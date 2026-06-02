# Notes for optimization:
#   * MAD may become extremely small on empty fields or diffims
#   * Could add an adaptive gamma based off of percentile
#   * Filtering of regions may be better using estimated fit RMS/aspect ratio
#   * Computing bottleneck is the Hessian matrix per sigma due to image size
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation
from scipy.stats import median_abs_deviation
from skimage.filters import frangi
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

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
        default=5.0,
    )
    min_vesselness = Field(
        doc="Minimum vesselness for threshold calculation.",
        dtype=float,
        default=1e-6,
    )
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
            return Struct(streak_catalog=streak_catalog, vesselness=vesselness)

        binary = vesselness > threshold
        binary[invalid] = False  # Future proof if threshold logic changes
        binary = remove_small_objects(binary, max_size=self.config.min_area)

        # Do simple pixel connectivity to generate an approximate line segment parametrization
        labels = label(binary, connectivity=2) 
        for region in regionprops(labels):
            if region.eccentricity < self.config.eccentricity:
                continue

            coords = region.coords
            ys = coords[:, 0]
            xs = coords[:, 1]
            weights = vesselness[ys, xs]
            fit_result = fit_line_segment_from_xy(xs, ys, weights=weights)

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

        return Struct(streak_catalog=catalog, vesselness=vesselness)
