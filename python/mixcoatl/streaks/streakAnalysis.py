from astropy.table import Table, Column

from lsst.afw.image import ImageF, MaskX
from lsst.pex.config import ChoiceField, ConfigurableField
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base.connectionTypes import Input, Output


class StreakAnalysisConnections(PipelineTaskConnections, dimensions=("instrument", "visit", "detector")):
    exposure = Input(
        doc="Background-subtracted exposure to detect streaks on.",
        name="preliminary_visit_image",
        storageClass="Exposure",
        dimensions=("instrument", "visit", "detector"),
    )
    detected_lines = Output(
        doc="Lines detected in the input exposure.",
        name="detected_lines",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "visit", "detector"),
    )

    # hessian exclusive outputs
    minima_ridges = Output(
        doc="Hessian matrix minima ridges image.",
        name="minima_ridges",
        storageClass="Image",
        dimensions=("instrument", "visit", "detector"),
    )
    binary_ridges = Output(
        doc="Detected ridges image.",
        name="binary_ridges",
        storageClass="Mask",
        dimensions=("instrument", "visit", "detector"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if config.detection_algorithm != "hessian":
            del self.minima_ridges
            del self.binary_ridges


class StreakAnalysisConfig(PipelineTaskConfig, pipelineConnections=StreakAnalysisConnections):
    detection_algorithm = ChoiceField(
        dtype=str,
        default="hessian",
        doc="Line detection algorithm to use.",
        allowed={"hessian": "Hessian matrix."},
    )
    detect_streaks = ConfigurableField(
        target=HessianDetectTask,
        doc="Detect streaks using Hessian matrix line detection.",
    )


class StreakAnalysisTask(PipelineTask):
    """Detect and measure linear features in an image."""
    ConfigClass = StreakAnalysisConfig
    _DefaultName = "streakAnalysis"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("detect_streaks")

    def run(self, exposure):

        result = Struct(detected_lines=None, minima_ridges=None, binary_ridges=None)

        if self.config.detection_algorithm == "hessian":
            detect_streak_result = self.detect_streaks.run(exposure)
            lines = detect_streak_result.lines
            result.minima_ridges = ImageF(detect_streak_result.minima_ridges.astype(np.float32))
            result.binary_ridges = MaskX(detcted_streak_result.binary_ridges.astype(np.int32))

        catalog = makeStreakCatalog()
        detector_id = exposure.getDetector().getId()
        for line in lines:
            rec = catalog.addNew()
            rec["detector"] = detector_id
            rec["line_rho"] = line.rho
            rec["line_theta"] = line.theta

        result.detected_lines = catalog

        return result
