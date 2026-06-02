import numpy as np

import lsst.afw.table import afwTable
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

from mixcoatl.streaks.table import StreakSchema

class RadonDetectConfig(pexConfig.Config):

    bad_mask_planes = pexConfig.ListField(
        doc="Names of mask plane regions to ignore when doing streak detection.",
        dtype=str,
        default=["NO_DATA", "INTRP", "BAD", "SAT", "EDGE", "ITL_DIP"],
    )
    npix_to_dilate = pexConfig.Field(
        doc="Number of pixels to dilate the bad mask planes.",
        dtype=int,
        default=8,
    )
    sigmas = pexConfig.ListField(
        doc="PSF scale for multiscale match filtering.",
        dtype=float,
        default=[1.0],
    )


class RadonDetectTask(pipeBase.Task):
    ConfigClass = RadonDetectConfig
    _DefaultName = "radonDetect"

    def run(self, exposure):
       
        schema = StreakSchema.makeMinimalSchema()
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
        image[grown_invalid] = 0.0 # proper masking for ADRT?

        # Convolve with PSF
        #getPSF
        #convolve with image

        # Format array for adrt

        # rad = adrt.adrt(convolved_image)

        # normalize radon response by noise acumulation (propagate variance analytically)

        # Convert radon peaks to line models
        # 

        return pipeBase.Struct(
            streak_catalog=catalog,
        )
