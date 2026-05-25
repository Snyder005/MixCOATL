# Can return Line2D plus a line segment representation (center/half-length), plus line fit statistics.
import numpy as np
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

import lsst.afw.math as afwMath
import lsst.geom as geom
from lsst.pex.config import Config, Field
from lsst.pipe.base import Struct, Task
from mixcoatl.streaks.line import fit_line_from_xy


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
    bin_size = Field(
        doc="Size of superpixel bins.",
        dtype=int,
        default=4
    )
    sigma = Field(
        doc="Size of the Gaussian kernel sigma used to compute the Hessian second derivatives.",
        dtype=float,
        default=12.0,
    )
    threshold_nsig = Field(
        doc="Threshold number of sigma applied to the Hessian minima ridges eigenvalue array.",
        dtype=float,
        default=5.0,
    )
    eccentricity = Field(
        doc="Lower bound for the eccentricity of the the minima regions.",
        dtype=float,
        default=0.98,
    )
    min_area = Field(
        doc="Mininum pixel area of minima regions.",
        dtype=int,
        default=5,
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

        # Bin exposure image and mask
        bin_size = self.config.bin_size
        masked_image = exposure.getMaskedImage()
        binned_image = afwMath.binImage(masked_image, bin_size)
        binned_array = binned.image.array.astype(np.float32, copy=False)        
        binned_mask = binned.mask.array

        # Suppress invalid pixels from bad mask planes
        bad_mask_bits = exposure.mask.getPlaneBitMask(self.config.bad_mask_planes)
        invalid = (binned_mask & bad_mask_bits) != 0
        binned_array = binned_array.copy()
        binned[invalid] = 0.0

        # Calculate hessian matrix
        h_elems = hessian_matrix(
            binned_array,
            sigma=self.config.sigma,
            order='rc',
            use_gaussian_derivatives=True,
        )
        maxima_ridges, minima_ridges = hessian_matrix_eigvals(h_elems)

        # Calculate binary array by thresholding
        valid_ridges = minima_ridges[~invalid]
        mad = median_abs_deviation(valid_ridges, scale="normal")
        threshold = np.median(valid_ridges) - self.config.threshold_nsig * mad
        binary_array = minima_ridges < threshold
        binary_array[invalid] = False

        lines = []
        labels = label(binary_array, connectivity=2)
        for region in regionprops(labels):

            # Skip tiny regions
            if region.area < self.config.min_area:  # add new configuration
                continue

            if region.eccentricity < self.config.eccentricity: # add new configuration
                continue

            coords = region.coords
            ys = coords[:, 0]
            xs = coords[:, 1]

            weights = np.abs(minima_ridges[ys, xs])
            line = fit_line_from_xy(xs, ys, weights=weights)

            transform = geom.AffineTransform(
                geom.LinearTransform.makeScaling(bin_size),
                geom.Extent2D(0.5 * (bin_size - 1), 0.5 * (bin_size - 1)),
            )

            lines.append(line.transformed(transform))

        return Struct(
            lines=lines,
            minima_ridges=minima_ridges,
            binary_ridges=binary_ridges,
        )
