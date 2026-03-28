import numpy as np
from astropy.table import Table, Column, vstack

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.daf.base as dafBase
import lsst.pipe.base.connectionTypes as cT

from lsst.pipe.tasks.coaddBase import reorderRefs

class ConsolidateStreaksConnections(pipeBase.PipelineTaskConnections, 
                                    dimensions=("instrument", "visit")):
    streaks = cT.Input(
        doc="Input per-detector streaks info.",
        name="streaks",
        storageClass="ArrowNumpyDict",
        dimensions=("instrument", "visit", "detector"),
        deferLoad=True,
        multiple=True,
    )
    streaks_visit = cT.Output(
        doc="Per-visit consolidation of streaks info.",
        name="streaks_visit",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "visit"),
    )

class ConsolidateStreaksConfig(pipeBase.PipelineTaskConfig,
                               pipelineConnections=ConsolidateStreaksConnections):
    pass

class ConsolidateStreaksTask(pipeBase.PipelineTask):
    """Consolidate `streaks` list into a per-visit `streaks_visit` table.
    """
    _DefaultName = "consolidateStreaks"
    ConfigClass = ConsolidateStreaksConfig

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        detectorOrder = [ref.dataId['detector'] for ref in inputRefs.streaks]
        detectorOrder.sort()
        inputRefs = reorderRefs(inputRefs, detectorOrder, dataIdKey="detector")

        handles = butlerQC.get(inputRefs.streaks)
        visit = handles[0].dataId['visit']
        self.log.info(f"Concatenating %s per-detector streaks dictionaries",
                      len(handles))
        results = self.run(visit=visit, handles=handles)
        butlerQC.put(results, outputRefs)

        handles = butlerQC.get(inputRefs.streaks)
        visit = handles[0].dataId['visit']

        results = self.run(visit=visit, handles=handles)

        butlerQC.put(results, outputRefs)

    def run(self, *, visit, handles):
        # Define table metadata
        visit_metadata = {'visit' : visit}
        
        # Define table columns
        table_visit = Table(meta=visit_metadata)

        for dataRef  in handles:
            streaks = dataRef.get()
            num_streaks = len(streaks['rho'])
            detector = dataRef.dataId['detector']

            table = Table()
            columns = [Column(name='detector', data=num_streaks*[detector])] + \
                      [Column(name=key, data=streaks[key]) for key in streaks]
            table.add_columns(columns)
    
            table_visit = vstack([table_visit, table])       

        result = pipeBase.Struct(streaks_visit=table_visit)
        return result
