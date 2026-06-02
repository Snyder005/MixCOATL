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
        storageClass="ArrowAstropy",  # what storageClass?
        dimensions=("instrument", "visit", "detector"),
    )

    # Frangi exclusive outputs
    vesselness = Output(
        doc="Frangi vesselness image.",
        name="vesselness",
        storageClass="Image",
        dimensions=("instrument", "visit", "detector"),
    )

    # Hessian exclusive outputs
    minima_ridges = Output(
        doc="Hessian matrix minima ridges image.",
        name="minima_ridges",
        storageClass="Image",
        dimensions=("instrument", "visit", "detector"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if config.detection_algorithm != "hessian":
            del self.minima_ridges
        if config.detection_algorithm !- "frangi":
            del self.vesselness


class StreakAnalysisConfig(PipelineTaskConfig, pipelineConnections=StreakAnalysisConnections):
    detection_algorithm = ChoiceField(
        dtype=str,
        default="hessian",
        doc="Line detection algorithm to use.",
        allowed={
            "hessian": "Hessian matrix.",
            "frangi" : "Multiscale Frangi vesselness.",
            "kht" : "Kernel Hough transform.",
            "radon" : "Matched filter Radon transform.",
        },
    )
    frangi_detect = ConfigurableField(
        target=FrangiDetectTask,
        doc="Detect streaks using multiscale Frangi vesselness.",
    )
    hessian_detect = ConfigurableField(
        target=HessianDetectTask,
        doc="Detect streaks using Hessian matrix.",
    )


class StreakAnalysisTask(PipelineTask):
    """Detect and measure linear features in an image."""
    ConfigClass = StreakAnalysisConfig
    _DefaultName = "streakAnalysis"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("detect_streaks")

    def run(self, exposure):

        if self.config.detection_algorithm == "hessian":
            detect_result = self.hessian_detect.run(exposure)
            return Struct(
                streak_catalog=streak_catalog,
                minima_ridges=ImageF(detect_result.minima_ridges.astype(np.float32)),
            )

        elif self.config.detection_algorithm == "frangi":
            detect_result = self.frangi_detect.run(exposure)
            return Struct(
                streak_catalog=streak_catalog,
                vesselness=ImageF(detect_result.vesselness.astype(np.float32)),
            )
