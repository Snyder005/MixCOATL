import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt
from skimage.feature import canny
from sklearn.cluster import KMeans

import lsst.afw.detection as afwDetect
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.geom as geom
import lsst.kht
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.meas.algorithms.maskStreaks import Line, LineProfile
from mixcoatl.streaks.line import Line2D
from mixcoatl.streaks.table import StreakAdapter


def get_pixel_mask(mask: afwImage.Mask, mask_plane: str | list[str]) -> NDArray[bool]:
    """Get pixel mask array corresponding to the mask planes.

    Parameters
    ----------
    mask : `lsst.afw.image.Mask`
        The input mask.
    mask_plane : `str` or `list` [`str`]
        Name or list of names of the mask plane.

    Returns
    -------
    pixel_mask : `numpy.ndarray`, (Ny, Nx)
        The pixel mask array.
    """
    pixel_mask = (mask.array & mask.getPlaneBitMask(mask_plane)) != 0
    return pixel_mask


def binary_dilation(binary_image: NDArray[bool], npix_to_dilate: int) -> NDArray[bool]:
    """Dilate a binary array with a circular structuring element.

    Parameters
    ----------
    binary_image : `numpy.ndarray`, (Ny, Nx)
        The input binary image array.
    npix_to_dilate : `int`
        The pixel radius of the circular structuring element to dilate by.

    Returns
    -------
    dilated_image : `numpy.ndarray`, (Ny, Nx)
        The dilated binary image array.
    """
    return distance_transform_edt(~binary_image) <= npix_to_dilate


class KHTDetectConfig(pexConfig.Config):
    """Configurable parameters for `KHTDetectTask`."""
    # Configuration for pixel masking
    detected_mask_plane = pexConfig.Field(
        doc="Name of mask plane with pixels above detection threshold.",
        dtype=str,
        default="DETECTED",
    )
    bad_mask_planes = pexConfig.ListField(
        doc="Names of mask plane regions to ignore when doing streak detection.",
        dtype=str,
        default=["NO_DATA", "INTRP", "BAD", "SAT", "EDGE", "ITL_DIP", "SPIKE"],
    )
    saturated_detections_dilation = pexConfig.Field(
        doc="Number of pixels to dilate the saturated detections mask.",
        dtype=int,
        default=250,
    )

    # Configuration for KHT
    cluster_minimum_size = pexConfig.Field(
        doc="Minimum size (in pixels) of detected clusters.",
        dtype=int,
        default=50,
    )
    cluster_minimum_deviation = pexConfig.Field(
        doc="Allowed deviation (in pixels) from a straight line.",
        dtype=int,
        default=2,
    )
    delta = pexConfig.Field(
        doc="Stepsize in angle-radius parameter space.",
        dtype=float,
        default=0.2,
    )
    minimum_kernel_height = pexConfig.Field(
        doc="Minimum height of the streak-finding kernel relative to the tallest kernel.",
        dtype=float,
        default=0.0,
    )
    nsigma = pexConfig.Field(
        doc="Number of sigma from center of kernel to include in voting procedure.",
        dtype=float,
        default=2.0,
    )
    abs_minimum_kernel_height = pexConfig.Field(
        doc="Minimum absolute height of the streak-finding kernel.",
        dtype=float,
        default=5.0,
    )

    # Configuration for clustering
    rho_bin_size = pexConfig.Field(
        doc="Binsize (in pixels) for position parameter when finding clusters.",
        dtype=float,
        default=40.0,
    )
    theta_bin_size = pexConfig.Field(
        doc="Binsize (in degrees) for angle parameter when finding clusters.",
        dtype=float,
        default=2.0,
    )

    # Configuration for profile fit
    inv_sigma = pexConfig.Field(
        doc="Inverse of the Moffat sigma parameter (in pixels) describing the streak profile.",
        dtype=float,
        default=10.**-1,
    )
    dchi2_tolerance = pexConfig.Field(
        doc="Absolute difference in chi2 between fit iterations for convergence.",
        dtype=float,
        default=0.1,
    )
    max_fit_iter = pexConfig.Field(
        doc="Maximum number of fit iterations acceptable for convergence.",
        dtype=int,
        default=100,
    )
    max_streak_width = pexConfig.Field(
        doc="Maximum width (in pixels) of the streak mask.",
        dtype=float,
        default=0.0,
    )
    nsigma_mask = pexConfig.Field(
        doc="Number of sigma from center of kernel to mask.",
        dtype=float,
        default=5.0,
    )
    footprint_threshold = pexConfig.Field(
        doc="Threshold at which to determine the edge of line (in nanoJansky).",
        dtype=float,
        default=0.01,
    )


class KHTDetectTask(pipeBase.Task):
    ConfigClass = KHTDetectConfig
    _DefaultName = "khtDetect"

    def run(self, table: afwTable.SourceTable, exposure: afwImage.ExposureF) -> pipeBase.Struct:

        image = exposure.image.array
        variance = exposure.variance.array
        mask = exposure.mask
        streaks = afwTable.SourceCatalog(table)
        wcs = exposure.getWcs()

        # Calculate Canny edge response
        detected_mask = get_pixel_mask(mask, self.config.detected_mask_plane)
        init_edges = canny(detected_mask.astype(np.float64), use_quantiles=True, sigma=0.1)

        # Mask invalid regions
        bad_mask = binary_dilation(get_pixel_mask(mask, self.config.bad_mask_planes), 1)
        if self.config.saturated_detections_dilation:
            sat_mask = get_pixel_mask(mask, "SAT")
            sat_detected_mask = binary_dilation(
                sat_mask & detected_mask,
                self.config.saturated_detections_dilation,
            )
            invalid = sat_detected_mask | bad_mask
        else:
            invalid = bad_mask
        edges = init_edges & ~invalid

        # Run KHT
        lines = lsst.kht.find_lines(
            edges,
            self.config.cluster_minimum_size,
            self.config.cluster_minimum_deviation,
            self.config.delta,
            self.config.minimum_kernel_height,
            self.config.nsigma,
            self.config.abs_minimum_kernel_height,
        )
        if lines.size == 0:
            return pipeBase.Struct(
                streak_catalog=catalog,
                detected=edges,
            )

        rhos, thetas = self._cluster_lines(lines.rho, lines.theta)
        
        # Fit Profile
        weights = variance**-1
        weights[~np.isfinite(weights) | ~np.isfinite(image)] = 0
        weights[bad_mask] = 0

        box = exposure.detector.getBBox()
        shift = box.getCenter().asExtent()
        for rho, theta in np.nditer((rhos, thetas)):
            line = Line(rho, theta, sigma=self.config.inv_sigma**-1)
            line_model = LineProfile(image, weights, line=line, detectionMask=detected_mask)
            if line_model.modelFailure or line_model.lineMask.sum() == 0:
                continue

            fit, failure = line_model.fit(self.config.dchi2_tolerance, maxIter=self.config.max_fit_iter)

            if ((abs(fit.rho - line.rho) > 2 * self.config.rho_bin_size) or 
                (abs(fit.theta - line.theta) > 2 * self.config.theta_bin_size)):
                failure = True

            if failure:
                continue

            line_model.setLineMask(fit, self.config.max_streak_width, self.config.nsigma_mask)
            final_model = line_model.makeProfile(fit)
            final_line_mask = abs(final_model) > self.config.footprint_threshold
            if not final_line_mask.any():
                continue

            kht_line = Line2D(fit.rho, fit.theta * geom.degrees)
            det_line = kht_line.translated(shift)
            line_segment = det_line.intersection(box)
            center = line_segment.center

            footprint_mask = afwImage.Mask(final_line_mask.astype(np.int32))
            spans = afwGeom.SpanSet.fromMask(footprint_mask)
            footprint = afwDetect.Footprint(spans)

            streak = StreakAdapter(streaks.addNew())
            streak.setLineSegment(line_segment)
            streak.setFootprint(footprint)
            streak.setCoord(wcs.pixelToSky(center))
            streak["line_center_x"] = center.getX()
            streak["line_center_y"] = center.getY()

        return pipeBase.Struct(
            streaks=streaks,
            edges=edges,
        )

    def _cluster_lines(
        self,
        rhos: NDArray[np.float64],
        thetas: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Cluster lines by recursive k-means clustering."""
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
        
        rhos = final_clusters[0] * self.config.rho_bin_size
        thetas = final_clusters[1] * self.config.theta_bin_size

        return rhos, thetas
