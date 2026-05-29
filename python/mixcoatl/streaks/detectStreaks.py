import numpy as np
from scipy.ndimage import binary_dilation
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

import lsst.afw.math as afwMath
import lsst.geom as geom
from lsst.pex.config import Config, Field
from lsst.pipe.base import Struct, Task
from mixcoatl.streaks.line import fit_line_segment_from_xy
from mixcoatl.streaks.table import StreakAdapter, StreakSchema

class KHTDetectConfig(Config):
    ...

class KHTDetectTask(Task):
    ConfigClass = KHTDetectConfig
    _DefaultName = "khtDetect"

    def run(self):
        ...


class HessianDetectConfig(Config):
    """Configurable parameters for StreakFinderTask.
    """
    sigma = Field(
        doc="Size of the Gaussian kernel sigma used to compute the Hessian second derivatives.",
        dtype=float,
        default=3.0,
    )
    threshold_nsig = Field(
        doc="Threshold number of sigma applied to the Hessian minima ridges eigenvalue array.",
        dtype=float,
        default=4.0,
    )
    eccentricity = Field(
        doc="Lower bound for the eccentricity of the the minima regions.",
        dtype=float,
        default=0.98,
    )
    min_area = Field(
        doc="Mininum pixel area of minima regions.",
        dtype=int,
        default=64,
    )
    bad_mask_planes = ListField(
        doc="Names of mask plane regions to ignore when doing streak detection.",
        dtype=str,
        default=("NO_DATA", "INTRP", "BAD", "SAT", "EDGE", "ITL_DIP"),
    )


class HessianDetectTask(Task):
    ConfigClass = HessianDetectConfig
    _DefaultName = "hessianDetect"

    def run(self, exposure):

        schema = StreakSchema.makeMinimalSchema()
        table = afwTable.SourceTable.make(schema)
        catalog = afwTable.SourceCatalog(table)

        masked_image = exposure.getMaskedImage()
        image = masked_image.image.array.astype(np.float32, copy=False)
        variance = masked_image.variance.array.astype(np.float32, copy=False)
        mask = masked_image.mask.array

        # Variance whitening
        image /= np.sqrt(np.maximum(variance, 1e-6)

        # Suppress invalid pixels from bad mask planes
        bad_mask_bits = exposure.mask.getPlaneBitMask(self.config.bad_mask_planes)
        invalid = (mask & bad_mask_bits) != 0
        grown_invalid = binary_dilation(invalid, iterations=max(1, int(np.ceil(2 * self.config.sigma))))

        vesselness = frangi(
            image,
            sigmas=self.config.sigmas,
            alpha=self.config.alpha,
            beta=self.config.beta,
            gamma=self.config.gamma,
            black_ridges=False,
        ) # Need all alpha, beta, gamma for 2-d?

        # Calculate binary array by thresholding
        valid = vesselness[~grown_invalid]
        if valid_ridges.size == 0:
            return Struct(
                streak_catalog=catalog,
                vesselness=vesselness,
            )

        mad = median_abs_deviation(valid, scale="normal")
        if mad <= 0 or not np.isfinite(mad):
            mad = np.std(valid_ridges)

        threshold = np.median(valid) + self.config.threshold_nsig * mad
        binary = vesselness > threshold
        binary[grown_invalid] = False # needed?

        # Do simple pixel connectivity to generate an approximate line segment parametrization
        labels = label(binary, connectivity=2)
        for region in regionprops(labels):

            if region.area < self.config.min_area:
                continue

            if region.eccentricity < self.config.eccentricity: # Better quantity to use (theta mean/std)
                continue

            coords = region.coords
            ys = coords[:, 0]
            xs = coords[:, 1]
            if np.any(grown_invalid[ys, xs]):
                continue

            weights = vesselness[ys, xs]
            line_segment = fit_line_segment_from_xy(xs, ys, weights=weights)

            span_list: list[afwGeom.Span] = []
            for y in np.unique(ys):
                x_row = xs[ys == y]
                span_list.append(afwGeom.Span(int(y), int(x_row.min()), int(x_row.max())))

            spans = afwGeom.SpanSet(span_list)
            footprint = afwDetect.Footprint(spans)

            record = catalog.addNew()
            streak = StreakAdapter(record)
            streak.line_segment = line_segment
            streak["detector"] = exposure.detector.getId()
            streak.footprint = footprint # may change API to setFootprint to match SourceRecord

            # May propogate in future (but can be calculated from Footprint)
            #streak["theta_mean"] = theta # calculated from Ixx, Iyy, Ixy of hessian matrix at fp center
            #streak["theta_std"] = 0.0
            #streak["ridge_strength"] = np.median(weights) # ridge strength
            #streak["ridge_width"] = 2.355 * orientation_sigma # ridge width estimate

        return Struct(
            streak_catalog=catalog,
            vesselness=vesselness,
        )
