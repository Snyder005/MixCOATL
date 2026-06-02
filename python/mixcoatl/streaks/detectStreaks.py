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

import lsst.kht
from lsst.afw.detection import Footprint
from lsst.afw.geom import SpanSet
from lsst.afw.image import ExposureF, MaskX
from lsst.afw.table import SourceCatalog, SourceTable
from lsst.pex.config import Config, Field, ListField
from lsst.pipe.base import Struct, Task
from mixcoatl.streaks.line import fit_line_segment_from_xy
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


def get_robust_threshold(
    likelihood: NDArray[np.float32],
    min_likelihood: float = 1e-6,
    nsig: float = 5.0,
) -> float:
    """Calculate a robust threshold for the likelhood array."""
    valid = likelihood[likelihood > min_likelihood]
    if valid.size == 0:
        raise ValueError("no valid points in the likelihood map")

    mad = median_abs_deviation(valid, scale="normal")
    if mad <= 0 or not np.isfinite(mad): # alternatively: max(mad, min_mad)
        mad = np.std(valid)

    threshold = np.median(valid) + nsig * mad
    return threshold


class FrangiDetectConfig(Config):
    """Configurable parameters for `FrangiDetectTask`.
    """
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
    npix_to_dilate = Field(
        doc="Number of pixels to dilate the bad mask planes.",
        dtype=int,
        default=8,
    )
    threshold_nsig = Field(
        doc="Number of sigma for vesselness thresholding.",
        dtype=float,
        default=5.0,
    )
    min_vesselness = Field(
        doc="Minimum vesselness for threshold calculation.",
        dtype=float,
        default=1e-6,
    )
    eccentricity = Field(
        doc="Lower bound for the eccentricity of the detected regions.",
        dtype=float,
        default=0.98,
    )
    min_area = Field(
        doc="Mininum pixel area of detected regions.",
        dtype=int,
        default=81,
    )
    bad_mask_planes = ListField(
        doc="Names of mask plane regions to ignore when doing streak detection.",
        dtype=str,
        default=["NO_DATA", "INTRP", "BAD", "SAT", "EDGE", "ITL_DIP"],
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

        # Variance whitening
        weighted_image = exposure.image.array / np.sqrt(np.maximum(exposure.variance.array, 1e-6))

        # Suppress invalid pixels from bad mask planes
        invalid = get_bad_pixel_mask(exposure, self.config.bad_mask_planes, self.config.npix_to_dilate)
        weighted_image[invalid] = 0.0

        vesselness = frangi(
            weighted_image,
            sigmas=self.config.sigmas,
            alpha=0.5,
            beta=self.config.beta,
            gamma=self.config.gamma,
            black_ridges=False,
        )
        vesselness[invalid] = 0.0

        # Calculate binary array by thresholding
        try:
            threshold = get_robust_threshold(
                vesselness,
                self.config.min_vesselness,
                self.config.threshold_nsig,
            )
        except:
            return Struct(streak_catalog=catalog, vesselness=vesselness)

        binary = vesselness > threshold
        binary[invalid] = False  # Future proof if threshold logic changes
        detected = remove_small_objects(detected, max_size=self.config.min_area)

        # Do simple pixel connectivity to generate an approximate line segment parametrization
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

        return Struct(streak_catalog=catalog, vesselness=vesselness, detected=detected)


class HessianDetectConfig(Config):
    """Configurable parameters for `HessianDetectClass`."""
    bad_mask_planes = ListField(
        doc="Names of mask plane regions to ignore when doing streak detection.",
        dtype=str,
        default=["NO_DATA", "INTRP", "BAD", "SAT", "EDGE", "ITL_DIP"],
    )
    npix_to_dilate = Field(
        doc="Number of pixels to dilate the bad mask planes.",
        dtype=int,
        default=8,
    )
    sigma = Field(
        doc="Gaussian scale for Hessian matrix.",
        dtype=float,
        default=2.0,
    )
    threshold_nsig = Field(
        doc="Number of sigma for Hessian thresholding.",
        dtype=float,
        default=4.0,
    )
    min_area = Field(
        doc="Minimum pixel area of detected regions.",
        dtype=int,
        default=81,
    )
    eccentricity = Field(
        doc="Lower bound for the eccentricity of the detected regions.",
        dtype=float,
        default=0.98,
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

        # Variance whitening
#        weighted_image = exposure.image.array / np.sqrt(np.maximum(exposure.variance.array, 1e-6))
        weighted_image = exposure.image.array.copy()

        # Suppress invalid pixels from bad mask planes
        invalid = get_bad_pixel_mask(exposure, self.config.bad_mask_planes, self.config.npix_to_dilate)
        weighted_image[invalid] = 0.0

        h_elems = hessian_matrix(
            weighted_image,
            sigma=self.config.sigma,
            order="rc",
            use_gaussian_derivatives=True,
        )
        maxima_ridges, minima_ridges = hessian_matrix_eigvals(h_elems)
        minima_ridges[invalid] = 0.0

        threshold = np.median(minima_ridges) - self.config.threshold_nsig * np.std(minima_ridges)
        binary = minima_ridges < threshold
        binary[invalid] = False
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

            if fit_result.aspect_ratio < 8 or not np.isfinite(fit_result.aspect_ratio):
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
            minima_ridges=minima_ridges,
            detected=detected,
        )


class KHTDetectConfig(Config):
    detected_mask_plane = Field(
        doc="Name of mask plane region with pixels above detection threshold.",
        dtype=str,
        default="DETECTED",
    )
    bad_mask_planes = ListField(
        doc="Names of mask plane regions to ignore when doing streak detection.",
        dtype=str,
        default=["NO_DATA", "INTRP", "BAD", "SAT", "EDGE", "ITL_DIP"],
    )
    npix_to_dilate = Field(
        doc="Number of pixels to dilate the bad mask planes.",
        dtype=int,
        default=8,
    )


class KHTDetectTask(Task):
    ConfigClass = KHTDetectConfig
    _DefaultName = "khtDetect"

    def run(self, exposure: ExposureF) -> Struct:

        schema = StreakSchema.makeMinimalSchema()
        table = afwTable.SourceTable.make(schema)
        catalog = afwTable.SourceCatalog(table)
        
        detected = exposure.mask.getPlaneBitMask(self.config.detected_mask_plane)
        weighted_image = (exposure.mask.array & detected)

        invalid = get_bad_pixel_mask(exposure, self.config.bad_mask_planes, self.config.npix_to_dilate)
        image[invalid] = 0.0

        init_edges = canny(image.astype(np.float64), use_quantiles=True, sigma=0.1)
        edges = init_edges & ~invalid  # This equivalent to vesselness or minima_ridges

        lines = lsst.kht.find_lines(
            edges,
            self.config.cluster_minimum_size,
            self.config.cluster_minimum_deviation,
            self.config.delta,
            self.config.minimum_kernel_height,
            self.config.nsigma,
            self.config.abs_minimum_kernel_height,
        )

        rhos = lines.rhos
        theta = lines.thetas

        box = exposure.detector.getBBox()
        shift = geom.Extent2D(-box.x, -box.y)

        # Cluster lines
        x = rhos / self.config.rho_bin_size
        y = thetas / self.config.theta_bin_size
        points = np.array([x, y]).T
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
        
        for rho, theta in np.nditer(final_rhos, final_thetas):

            kht_line = Line2D(rho=line.rho, theta=line.theta * geom.degrees)
            det_line = kht_line.translated(shift)
            
            record = catalog.addNew()
            streak = StreakAdapter(record)
            streak.line_segment = det_line.intersection(box)

        return Struct(streak_catalog=catalog, edges=edges)
