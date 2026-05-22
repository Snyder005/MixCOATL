import numpy as np
from astropy.table import Table, Column, vstack

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.daf.base as dafBase
import lsst.pipe.base.connectionTypes as cT

from lsst.pipe.tasks.coaddBase import reorderRefs

class ConsolidateGlintsConnections(pipeBase.PipelineTaskConnections, 
                                   dimensions=("instrument", "visit")):
    trailed_glints = cT.Input(
        doc="Input per-detector trailed glints info.",
        name="trailed_glints",
        storageClass="ArrowNumpyDict",
        dimensions=("instrument", "visit", "detector"),
        deferLoad=True,
        multiple=True,
    )
    trailed_glints_visit = cT.Output(
        doc="Per-visit consolidation of trailed glints info.",
        name="trailed_glints_visit",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "visit"),
    )

class ConsolidateGlintsConfig(pipeBase.PipelineTaskConfig,
                              pipelineConnections=ConsolidateGlintsConnections):
    pass

class ConsolidateGlintsTask(pipeBase.PipelineTask):
    """Consolidate `trailed_glint_info` list into a per-visit 
       `trailed_glint_visit`.
    """
    _DefaultName = "consolidateGlints"
    ConfigClass = ConsolidateGlintsConfig

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        detectorOrder = [ref.dataId['detector'] for ref in inputRefs.trailed_glints]
        detectorOrder.sort()
        inputRefs = reorderRefs(inputRefs, detectorOrder, dataIdKey="detector")

        handles = butlerQC.get(inputRefs.trailed_glints)
        visit = handles[0].dataId['visit']
        self.log.info(f"Concatenating %s per-detector trailed glint dictionaries",
                      len(handles))
        results = self.run(visit=visit, handles=handles)
        butlerQC.put(results, outputRefs)

    def run(self, *, visit, handles):
        """Make a trailed glints summary from a list of handles.
        These handles must point to trailed glint information dictionaries.

        Parameters
        ----------
        visit : `int`
            Visit identification number.
        handles: `list` of `lsst.daf.butler.DeferredDatasetHandles`
            List of handles in visit.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Struct with the following attributes:

            - ``trailed_glints_summary`` (`dict`): A per-visit dictionary with
              trailed glints information.
        """
        # Define table metadata
        visit_metadata = {'visit' : visit}
        
        # Define table columns
        table_visit = Table(meta=visit_metadata)

        for dataRef in handles:
            trailed_glints = dataRef.get()
            num_glints = len(trailed_glints['slopes'])
            detector = dataRef.dataId['detector']

            table = Table()
            columns = [Column(name='detector', data=num_glints*[detector])] + \
                      [Column(name=key, data=trailed_glints[key]) for key in trailed_glints]
            table.add_columns(columns)
    
            table_visit = vstack([table_visit, table])

        result = pipeBase.Struct(trailed_glints_visit=table_visit)
        return result
