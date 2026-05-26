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
        mask = masked_image.mask.array

        # Suppress invalid pixels from bad mask planes
        bad_mask_bits = exposure.mask.getPlaneBitMask(self.config.bad_mask_planes)
        invalid = (mask & bad_mask_bits) != 0
        grown_invalid = binary_dilation(invalid, iterations=max(1, int(np.ceil(2 * self.config.sigma))))

        # Calculate hessian matrix
        h_elems = hessian_matrix(
            image,
            sigma=self.config.sigma,
            order='rc',
            use_gaussian_derivatives=True,
        )
        maxima_ridges, minima_ridges = hessian_matrix_eigvals(h_elems)

        # Calculate binary array by thresholding
        valid_ridges = minima_ridges[~grown_invalid]
        if valid_ridges.size == 0:
            return Struct(
                streak_catalog=catalog,
                minima_ridges=minima_ridges,
                binary_ridges=np.zeros_like(minima_ridges, dtype=bool),
            )

        mad = median_abs_deviation(valid_ridges, scale="normal")
        if mad <= 0 or not np.isfinite(mad):
            mad = np.std(valid_ridges)

        threshold = np.median(valid_ridges) - self.config.threshold_nsig * mad
        binary = minima_ridges < threshold
        binary[grown_invalid] = False

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

            weights = np.abs(minima_ridges[ys, xs])
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
            streak["detector"] = exposure.info.detector.getId()
            streak.footprint = footprint

        return Struct(
            streak_catalog=catalog,
            minima_ridges=minima_ridges,
            binary_ridges=binary,
        )
