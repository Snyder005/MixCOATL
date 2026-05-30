# Notes for optimization:
#   * MAD may become extremely small on empty fields or diffims
#   * Could add an adaptive gamma based off of percentile
#   * Filtering of regions may be better using estimated fit RMS/aspect ratio
#   * Computing bottleneck is the Hessian matrix per sigma due to image size
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.stats import median_abs_deviation
from skimage.filters import frangi
from skimage.morphology import remove_small_objects

import lsst.afw.detection as afwDetect
import lsst.afw.geom as afwGeom
import lsst.afw.table as afwTable
from lsst.pex.config import Config, Field, ListField
from lsst.pipe.base import Struct, Task
from mixcoatl.streaks.line import fit_line_segment_from_xy
from mixcoatl.streaks.table import StreakAdapter, StreakSchema


class FrangiDetectConfig(Config):
    """Configurable parameters for StreakFinderTask.
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
        default=5.0
    eccentricity = Field(
        doc="Lower bound for the eccentricity of the the minima regions.",
        dtype=float,
        default=0.98,
    )
    min_area = Field(
        doc="Mininum pixel area of minima regions.",
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

    def run(self, exposure):

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

        table = afwTable.SourceTable.make(schema)
        catalog = afwTable.SourceCatalog(table)

        masked_image = exposure.getMaskedImage()
        image = masked_image.image.array.astype(np.float32, copy=True)
        variance = masked_image.variance.array.astype(np.float32, copy=True)
        mask = masked_image.mask.array

        # Variance whitening
        image /= np.sqrt(np.maximum(variance, 1e-6))

        # Suppress invalid pixels from bad mask planes
        bad_mask_bits = exposure.mask.getPlaneBitMask(self.config.bad_mask_planes)
        invalid = (mask & bad_mask_bits) != 0
        grown_invalid = binary_dilation(invalid, iterations=max(1, self.config.npix_to_dilate))
        image[grown_invalid] = 0.0

        vesselness = frangi(
            image,
            sigmas=self.config.sigmas,
            alpha=0.5,
            beta=self.config.beta,
            gamma=self.config.gamma,
            black_ridges=False,
        )
        vesselness[grown_invalid] = 0.0

        # Calculate binary array by thresholding
        valid = vesselness[vesselness > 1e-6] # alternatively: min_vesselness_floor
        if valid.size == 0:
            return Struct(
                streak_catalog=catalog,
                vesselness=vesselness,
            )

        mad = median_abs_deviation(valid, scale="normal")
        if mad <= 0 or not np.isfinite(mad): # alternatively: max(mad, self.config.min_mad)
            mad = np.std(valid)

        threshold = np.median(valid) + self.config.threshold_nsig * mad
        binary = vesselness > threshold
        binary[grown_invalid] = False
        binary = remove_small_objects(binary, max_size=self.config.min_area)

        # Do simple pixel connectivity to generate an approximate line segment parametrization
        labels = label(binary, connectivity=2) 
        for region in regionprops(labels):

            if region.area < self.config.min_area:
                continue

            if region.eccentricity < self.config.eccentricity:
                continue

            coords = region.coords
            ys = coords[:, 0]
            xs = coords[:, 1]
            if np.any(grown_invalid[ys, xs]):
                continue

            weights = vesselness[ys, xs]
            fit_result = fit_line_segment_from_xy(xs, ys, weights=weights)

            footprint_mask = np.zeros_like(binary, dtype=np.uint8)
            footprint_mask[ys, xs] = 1

            spans = afwGeom.SpanSet.fromMask(footprint_mask.astype(np.int32))
            footprint = afwDetect.Footprint(spans)

            record = catalog.addNew()
            streak = StreakAdapter(record)
            streak.line_segment = fit_result.line_segment
            streak["detector"] = exposure.detector.getId()
            streak["line_rms"] = fit_result.rms
            streak["aspect_ratio"] = fit_result.aspect_ratio
            streak.footprint = footprint

        return Struct(
            streak_catalog=catalog,
            vesselness=vesselness,
        )
