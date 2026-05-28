import numpy as np
from skimage.feature import canny

import lsst.afw.table as afwTable
import lsst.afw.geom as afwGeom
from lsst.pex.config import Config, Field, ListField
from lsst.pipe.base import Task
from mixcoatl.streaks.table import StreakAdapter, StreakSchema

class KHTDetectConfig(Config):
    
    detected_mask_plane = Field(
        doc="Name of mask with pixels above detection threshold.",
        dtype=str,
        default="DETECTED",
    )
    bad_mask_planes = ListField(
        doc="Names of mask plane regions to ignore when doing streak detection.",
        dtype=str,
        default=("NO_DATA", "INTRP", "BAD", "SAT", "EDGE", "ITL_DIP"),
    )

class KHTDetectTask(Task):
    ConfigClass = KHTDetectConfig
    _DefaultName = "khtDetect"

    def run(self, exposure):
        
        schema = StreakSchema.makeMinimalSchema()
        table = afwTable.SourceTable.make(schema)
        catalog = afwTable.SourceCatalog(table)

        masked_image = exposure.getMaskedImage()
        mask = exposure.mask
        detection_array = (mask.array & mask.getPlaneBitMask(self.config.detected_mask_plane))

        init_edges = canny((detection_array > 0).astype(np.float64), use_quantiles=True, sigma=0.1)

        ignore_mask = mask.clone()

        bad_bit_mask = mask.getPlaneBitMask(self.config.bad_mask_planes)
        bad_span_set = afwGeom.SpanSet.fromMask(mask, bad_bit_mask).split()
        for sset in bad_span_set:
            sset_dilated = sset.dilated(1)
            sset_dilated.clippedTo(ignore_mask.getBBox()).setMask(
                ignore_mask,
                ignore_mask.getPlaneBitMask("BAD"),
            )

        if self.config.saturated_detections_dilation:
            sat_array = (mask.array & mask.getPlaneBitMask("SAT"))
            sat_det_array = (sat_array != 0) & (detection_array != 0)
            sat_det_mask = afwImage.Mask(sat_det_array.astype(np.int32))
            sat_span_set = afwGeom.SpanSet.fromMask(sat_det_mask, 1).split()
            for sset in sat_span_set:
                sset_dilated = sset.dilated(self.config.saturated_detections_dilation)
                sset_dilated.clippedTo(ignore_mask.getBBox()).setMask(
                    ignore_mask,
                    ignore_mask.getPlaneBitMask("BAD"),
                )

        dilated_bad_array = (ignore_mask.array & bad_bit_mask) > 0
        edges = init_edges & ~dilated_bad_array

        lines = lsst.kht.find_lines(
            edges & ~dilated_bad_array,
            self.config.cluster_minimum_size,
            self.config.cluster_minimum_deviation,
            self.config.delta,
            self.config.minimum_kernel_height,
            self.config.nsigma,
            self.config.abs_minimum_kernel_height,
        )
        if len(lines) == 0:
            return Struct(streak_catalog=catalog)


        clusters = self._findClusters(lines) # per-detector clustering
        fit_lines, line_mask = self._fitProfile(clusters, masked_image, detectionMask=detection_array)

        for l in fit_lines:
            line = Line2D(l.rho, l.theta * geom.radians)
            # get line segment from detector bbox intersection
            # add record

        return Struct(
            streak_catalog=catalog,
            original_lines=lines,
            mask=line_mask,
        )
