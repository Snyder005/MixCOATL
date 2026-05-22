import numpy as np
from astropy.table import Table, Column, vstack

from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base.connectionTypes import Input, Output
from lsst.pipe.tasks.coaddBase import reorderRefs

__all__ = ["ConsolidateStreaksTask", "ConsolidateStreaksConfig"]


class ConsolidateStreaksConnections(PipelineTaskConnections, dimensions=("instrument", "visit")):
    streaks = Input(
        doc="Input per-detector streaks info.",
        name="streaks",
        storageClass="ArrowNumpyDict",
        dimensions=("instrument", "visit", "detector"),
        deferLoad=True,
        multiple=True,
    )
    streaks_visit = Output(
        doc="Per-visit consolidation of streaks info.",
        name="streaks_visit",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "visit"),
    )


class ConsolidateStreaksConfig(PipelineTaskConfig, pipelineConnections=ConsolidateStreaksConnections):
    pass


class ConsolidateStreaksTask(PipelineTask):
    """Consolidate `streaks` list into a per-visit `streaks_visit` table.
    """
    _DefaultName = "consolidateStreaks"
    ConfigClass = ConsolidateStreaksConfig

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        detectorOrder = [ref.dataId["detector"] for ref in inputRefs.streaks]
        detectorOrder.sort()
        inputRefs = reorderRefs(inputRefs, detectorOrder, dataIdKey="detector")

        handles = butlerQC.get(inputRefs.streaks)
        visit = handles[0].dataId["visit"]

        self.log.info(f"Concatenating {len(handles)} per-detector streaks dictionaries")
        results = self.run(visit=visit, handles=handles)
        butlerQC.put(results, outputRefs)

    def run(self, *, visit, handles):
        # Define table metadata
        visit_metadata = {"visit" : visit}
        
        # Define table columns
        table_visit = Table(meta=visit_metadata)

        for dataRef  in handles:
            streaks = dataRef.get()
            num_streaks = len(streaks["rho"])
            detector = dataRef.dataId["detector"]

            table = Table()
            detector_col = Column(name="detector", data=num_streaks * [detector], dtype="int32")
            streak_cols = [Column(name=key, data=value, dtype="int32") for key, value in streaks.items()]
            table.add_columns([detector_col] + streaks_cols)

            table_visit = vstack([table_visit, table])

        table_visit["detector"] = table_visit["detector"].astype("int32")

        result = Struct(streaks_visit=table_visit)
        return result
