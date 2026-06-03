# Notes for optimization:
#   * MAD may become extremely small on empty fields or diffims
#   * Could add an adaptive gamma based off of percentile
#   * Filtering of regions may be better using estimated fit RMS/aspect ratio
#   * Computing bottleneck is the Hessian matrix per sigma due to image size
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation
from scipy.stats import median_abs_deviation
from skimage.feature import canny, hessian_matrix, hessian_matrix_eigvals
from skimage.filters import frangi  # For Frangi
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from sklearn.cluster import KMeans

import lsst.kht
import lsst.geom as geom
from lsst.afw.detection import Footprint
from lsst.afw.geom import SpanSet
from lsst.afw.image import ExposureF, MaskX
from lsst.afw.table import SourceCatalog, SourceTable
from lsst.pex.config import Config, Field, ListField
from lsst.pipe.base import Struct, Task
from mixcoatl.streaks.line import fit_line_segment_from_xy, Line2D
from mixcoatl.streaks.table import StreakAdapter, StreakSchema


def get_bad_pixel_mask(
    exposure: ExposureF,
    bad_mask_planes: list[str],
    npix_to_dilate: int = 8,
) -> NDArray[np.bool]:
    """Get a bad pixel mask."""
    bad_mask_bits = exposure.mask.getPlaneBitMask(bad_mask_planes)
    bad = (exposure.mask.array & bad_mask_bits) != 0
    bad = binary_dilation(bad, iterations=max(1, npix_to_dilate))
    
    return bad


def calculate_response_significance(
    response: NDArray[np.float32],
    invalid: NDArray[np.bool],
    min_response: float | None = None,
) -> tuple[NDArray[np.float32], float, float]:

    valid = np.isfinite(response) & ~invalid
    if min_response is not None:
        valid &= (response > min_response)

    responses = response[valid]

    if responses.size == 0:
        raise ValueError("no valid points in the streak detector response")

    median = np.median(responses)
    sigma = median_abs_deviation(responses, scale="normal")
    if sigma <= 0 or not np.isfinite(sigma):
        sigma = np.std(valid)

    if sigma <= 0 or not np.isfinite(sigma):
        raise ValueError("streak detector response has no measurable dispersion")

    significance = (response - median) / sigma
    return significance, median, sigma


class FrangiDetectConfig(Config):
    """Configurable parameters for `FrangiDetectTask`."""
    # Configuration for masking
    bad_mask_planes = ListField(
        doc="Names of mask plane regions to ignore when doing streak detection.",
        dtype=str,
        default=["NO_DATA", "INTRP", "BAD", "SAT", "EDGE", "ITL_DIP", "SPIKE"],
    )
    npix_to_dilate = Field(
        doc="Number of pixels to dilate the bad mask planes.",
        dtype=int,
        default=8,
    )

    # Configuration for Frangi filter
    sigmas = ListField(
        doc="Gaussian scale for multiscale Hessian matrix.",
        dtype=float,
        default=[1.0, 2.0, 4.0],
    )
    beta = Field(
        doc="Frangi correction constant to adjust sensitivity to anisotropy.",
        dtype=float,
        default=0.5,
    )
    gamma = Field(
        doc="Frangi correction constant to adjust sensitivity to structural strength.",
        dtype=float,
        default=3.0,
    )

    # Configuration for response thresholding
    response_nsig = Field(
        doc="Number of sigma for streak detector response thresholding.",
        dtype=float,
        default=5.0,
    )
    min_vesselness = Field(
        doc="Minimum vesselness for threshold calculation.",
        dtype=float,
        default=1e-6,
    )

    # Configuration for detected regions
    min_area = Field(
        doc="Mininum pixel area of the detected regions.",
        dtype=int,
        default=81,
    )
    eccentricity = Field(
        doc="Lower bound for the eccentricity of the detected regions.",
        dtype=float,
        default=0.98,
    )


class FrangiDetectTask(Task):
    ConfigClass = FrangiDetectConfig
    _DefaultName = "frangiDetect"

    def run(self, exposure: ExposureF) -> Struct:

        # Create a streak catalog
        schema = StreakSchema.makeMinimalSchema()
        schema.addField(
            "line_rms",
            type=np.float32,
            doc="Weighted perpendicular RMS residual from line fit.",
        )
        schema.addField(
            "aspect_ratio",
            type=np.float32,
            doc="Length divided by estimated width.",
        )
        table = SourceTable.make(schema)
        catalog = SourceCatalog(table)

        # Calculate weighted image
        weighted_image = exposure.image.array / np.sqrt(np.maximum(exposure.variance.array, 1e-6))

        # Suppress invalid pixels from bad mask planes
        invalid = get_bad_pixel_mask(exposure, self.config.bad_mask_planes, self.config.npix_to_dilate)
        weighted_image[invalid] = 0.0

        # Calculate Frangi vesselness
        vesselness = frangi(
            weighted_image,
            sigmas=self.config.sigmas,
            alpha=0.5,
            beta=self.config.beta,
            gamma=self.config.gamma,
            black_ridges=False,
        )
        vesselness[invalid] = 0.0

        # Calculate background-normalized significance
        significance, median, sigma = calculate_response_significance(
            vesselness,
            invalid,
            min_response=self.config.min_vesselness,
        )
        binary = (significance > self.config.response_nsig)
        binary[invalid] = False

        # Connect pixel regions and add to catalog
        detected = remove_small_objects(binary, max_size=self.config.min_area)
        labels = label(detected, connectivity=2)
        for region in regionprops(labels):
            if region.eccentricity < self.config.eccentricity:
                continue

            coords = region.coords
            ys = coords[:, 0]
            xs = coords[:, 1]
            weights = vesselness[ys, xs]
            fit_result = fit_line_segment_from_xy(xs, ys, weights=weights)

            footprint_mask = MaskX(np.zeros_like(detected, dtype=np.int32))
            footprint_mask.array[ys, xs] = 1
            spans = SpanSet.fromMask(footprint_mask)
            footprint = Footprint(spans)

            record = catalog.addNew()
            streak = StreakAdapter(record)
            streak.line_segment = fit_result.line_segment
            streak["detector"] = exposure.detector.getId()
            streak["line_rms"] = fit_result.rms
            streak["aspect_ratio"] = fit_result.aspect_ratio
            streak.footprint = footprint

        return Struct(
            streak_catalog=catalog,
            significance=significance,
            detected=detected,
            response=vesselness,
        )


class HessianDetectConfig(Config):
    """Configurable parameters for `HessianDetectClass`."""
    # Configuration for masking
    bad_mask_planes = ListField(
        doc="Names of mask plane regions to ignore when doing streak detection.",
        dtype=str,
        default=["NO_DATA", "INTRP", "BAD", "SAT", "EDGE", "ITL_DIP", "SPIKE"],
    )
    npix_to_dilate = Field(
        doc="Number of pixels to dilate the bad mask planes.",
        dtype=int,
        default=8,
    )

    # Configuration for Hessian matrix
    sigma = Field(
        doc="Gaussian scale for Hessian matrix.",
        dtype=float,
        default=1.0,
    )

    # Configruation for response thresholding
    response_nsig = Field(
        doc="Number of sigma for thresholding of the streak detector response.",
        dtype=float,
        default=5.0,
    )

    # Configuration for detected regions
    min_area = Field(
        doc="Minimum pixel area of the detected regions.",
        dtype=int,
        default=81,
    )
    eccentricity = Field(
        doc="Lower bound for the eccentricity of the detected regions.",
        dtype=float,
        default=0.98,
    )
    aspect_ratio = Field(
        doc="Lower bound for the aspect ratio of the detected regions.",
        dtype=float,
        default=8.0,
    )


class HessianDetectTask(Task):
    ConfigClass = HessianDetectConfig
    _DefaultName = "hessianDetect"

    def run(self, exposure: ExposureF) -> Struct:

        # Create a streak catalog
        schema = StreakSchema.makeMinimalSchema()
        schema.addField(
            "line_rms",
            type=np.float32,
            doc="Weighted perpendicular RMS residual from line fit.",
        )
        schema.addField(
            "aspect_ratio",
            type=np.float32,
            doc="Length divided by estimated width.",
        )
        table = SourceTable.make(schema)
        catalog = SourceCatalog(table)

        # Calculate weighted image
        weighted_image = exposure.image.array / np.sqrt(np.maximum(exposure.variance.array, 1e-6))

        # Suppress invalid pixels from bad mask planes
        invalid = get_bad_pixel_mask(exposure, self.config.bad_mask_planes, self.config.npix_to_dilate)
        weighted_image[invalid] = 0.0

        # Calculate Hessian eigenvalues
        h_elems = hessian_matrix(
            weighted_image,
            sigma=self.config.sigma,
            order="rc",
            use_gaussian_derivatives=True,
        )
        h_elems = [self.config.sigma**2 * h for h in h_elems]
        maxima_ridges, minima_ridges = hessian_matrix_eigvals(h_elems)
        response = -minima_ridges
        response[invalid] = 0.0

        # Calculate background-normalized significance
        significance, median, sigma = calculate_response_significance(response, invalid)
        binary = significance > self.config.response_nsig
        binary[invalid] = False

        # Connect pixel regions and add to catalog
        detected = remove_small_objects(binary, max_size=self.config.min_area)
        labels = label(detected, connectivity=2)
        for region in regionprops(labels):
            if region.eccentricity < 0.98:
                continue

            coords = region.coords
            ys = coords[:, 0]
            xs = coords[:, 1]
            weights = detected[ys, xs]
            fit_result = fit_line_segment_from_xy(xs, ys, weights=weights)

            if fit_result.aspect_ratio < self.config.aspect_ratio:
                continue
            
            if not np.isfinite(fit_result.aspect_ratio):
                continue

            footprint_mask = MaskX(np.zeros_like(binary, dtype=np.int32))
            footprint_mask.array[ys, xs] = 1
            spans = SpanSet.fromMask(footprint_mask)
            footprint = Footprint(spans)

            record = catalog.addNew()
            streak = StreakAdapter(record)
            streak.line_segment = fit_result.line_segment
            streak["detector"] = exposure.detector.getId()
            streak["line_rms"] = fit_result.rms
            streak["aspect_ratio"] = fit_result.aspect_ratio
            streak.footprint = footprint

        return Struct(
            streak_catalog=catalog,
            significance=significance,
            detected=detected,
            response=response,
        )


class KHTDetectConfig(Config):
    """Configurable parameters for `KHTDetectTask`."""
    # Configuration for masking
    detected_mask_plane = Field(
        doc="Name of mask plane region with pixels above detection threshold.",
        dtype=str,
        default="DETECTED",
    )

    # Configuration for masking
    bad_mask_planes = ListField(
        doc="Names of mask plane regions to ignore when doing streak detection.",
        dtype=str,
        default=["NO_DATA", "INTRP", "BAD", "SAT", "EDGE", "ITL_DIP", "SPIKE"],
    )
    npix_to_dilate = Field(
        doc="Number of pixels to dilate the bad mask planes.",
        dtype=int,
        default=8,
    )

    # Configuration for Kernel Hough transform
    cluster_minimum_size = Field(
        doc="Minimum size (in pixels) of detected clusters.",
        dtype=int,
        default=50,
    )
    cluster_minimum_deviation = Field(
        doc="Allowed deviation (in pixels) from a straight line for a detected line.",
        dtype=int,
        default=2,
    )
    delta = Field(
        doc="Stepsize in angle-radius parameter space.",
        dtype=float,
        default=0.2,
    )
    minimum_kernel_height = Field(
        doc="Minimum height of the streak-finding kernel relative to the tallest kernel.",
        dtype=float,
        default=0.0,
    )
    nsigma = Field(
        doc="Number of sigma from center of kernel to include in voting procedure.",
        dtype=float,
        default=2.0,
    )
    abs_minimum_kernel_height = Field(
        doc="Minimum absolute height of the streak-finding kernel.",
        dtype=float,
        default=5.0,
    )

    # Configuration for clustering
    rho_bin_size = Field(
        doc="Binsize (in pixels) for position parameter when finding clusters of detected lines.",
        dtype=float,
        default=40.0,
    )
    theta_bin_size = Field(
        doc="Binsize (in degrees) for angle parameter theta when finding clusters of detected lines.",
        dtype=float,
        default=2.0,
    )


class KHTDetectTask(Task):
    ConfigClass = KHTDetectConfig
    _DefaultName = "khtDetect"

    def run(self, exposure: ExposureF) -> Struct:

        # Create a streak catalog
        schema = StreakSchema.makeMinimalSchema()
        table = SourceTable.make(schema)
        catalog = SourceCatalog(table)
        
        # Calculate weighted image
        detected = exposure.mask.getPlaneBitMask(self.config.detected_mask_plane)
        weighted_image = (exposure.mask.array & detected)

        # Suppress invalid pixels from bad mask planes
        invalid = get_bad_pixel_mask(exposure, self.config.bad_mask_planes, self.config.npix_to_dilate)

        # Calculate Canny edge response
        edges = canny(weighted_image.astype(np.float64), use_quantiles=True, sigma=0.1)
        edges[invalid] = False

        # Detect and cluster lines
        lines = lsst.kht.find_lines(
            edges,
            self.config.cluster_minimum_size,
            self.config.cluster_minimum_deviation,
            self.config.delta,
            self.config.minimum_kernel_height,
            self.config.nsigma,
            self.config.abs_minimum_kernel_height,
        )
        rhos, thetas = self._cluster_lines(lines.rho, lines.theta)

        box = exposure.detector.getBBox()
        shift = box.getCenter().asExtent()

        for rho, theta in np.nditer((rhos, thetas)):
            kht_line = Line2D(rho, theta * geom.degrees)
            det_line = kht_line.translated(shift)
            
            record = catalog.addNew()
            streak = StreakAdapter(record)
            streak.line_segment = det_line.intersection(box)

        return Struct(
            streak_catalog=catalog,
            detected=edges,
        )

    def _cluster_lines(
        self,
        rhos: NDArray[np.float64],
        thetas: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Cluster lines by recursive kmeans clustering."""
        x = rhos / self.config.rho_bin_size
        y = thetas / self.config.theta_bin_size
        points = np.column_stack((x, y))
        
        n_clusters = 1
        while True:
            kmeans = KMeans(n_clusters=n_clusters, n_init="auto").fit(points)
            cluster_standard_deviations = np.zeros((n_clusters, 2))
            for c in range(n_clusters):
                in_cluster = points[kmeans.labels_ == c]
                cluster_standard_deviations[c] = np.std(in_cluster, axis=0)
                
            if (cluster_standard_deviations <= 1).all():
                break
                
            n_clusters += 1
        
        final_clusters = kmeans.cluster_centers_.T
        
        final_rhos = final_clusters[0] * self.config.rho_bin_size
        final_thetas = final_clusters[1] * self.config.theta_bin_size

        return final_rhos, final_thetas
