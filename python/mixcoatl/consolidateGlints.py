import numpy as np

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.daf.base as dafBase
import lsst.pipe.base.connectionTypes as cT

class ConsolidateGlintsConnections(pipeBase.PipelineTaskConnections, 
                                   dimensions=("instrument", "visit")):
    trailed_glints = cT.Input(
        doc="Trailed glints info.",
        name="trailed_glints",
        storageClass="ArrowNumpyDict",
        dimensions=("instrument", "visit", "detector"),
        deferLoad=True,
        multiple=True,
    )
    trailed_glints_summary = cT.Output(
        doc="Consolidated trailed glints info.",
        name="trailed_glints_summary",
        storageClass="ArrowNumpyDict",
        dimensions=("instrument", "visit"),
    )

class ConsolidateGlintsConfig(pipeBase.PipelineTaskConfig,
                              pipelineConnections=ConsolidateGlintsConnections): pass

class ConsolidateGlintsTask(pipeBase.PipelineTask):
    _DefaultName = "consolidateGlints"
    ConfigClass = ConsolidateGlintsConfig

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        handles = butlerQC.get(inputRefs.trailed_glints)
        visit = handles[0].dataId["visit"]

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
        summary = {
            'visit' : [],
            'detector' : [],
            'slopes' : [],
            'intercepts' : [],
            'stderrs' : [],
            'lengths' : [],
            'angles' : []
        }
        for i, dataRef  in enumerate(handles):
            trailed_glints = dataRef.get()
            detector = dataRef.dataId['detector']
            num_glints = len(trailed_glints['slopes'])
            for j in range(num_glints):
                summary['visit'].append(visit)
                summary['detector'].append(detector)
                for k in trailed_glints:
                    summary[k].append(trailed_glints[k][j])

        for k in summary:
            summary[k] = np.asarray(summary[k])

        result = pipeBase.Struct(trailed_glints_summary=summary)
        return result
