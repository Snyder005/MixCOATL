import numpy as np

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.daf.base as dafBase
import lsst.pipe.base.connectionTypes as cT

class ConsolidateGlintsConnections(pipeBase.PipelineTaskConnections, dimensions=("instrument",)):
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
        dimensions=("instrument",),
    )

class ConsolidateGlintsConfig(pipeBase.PipelineTaskConfig,
                              pipelineConnections=ConsolidateGlintsConnections): pass

class ConsolidateGlintsTask(pipeBase.PipelineTask):
    _DefaultName = "consolidateGlints"
    ConfigClass = ConsolidateGlintsConfig

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        handles = butlerQC.get(inputRefs.trailed_glints)

        results = self.run(handles)

        butlerQC.put(results, outputRefs)

    def run(self, handles):
        
        summary = {'visit' : [],
                   'detector' : [],
                   'slopes' : [],
                   'intercepts' : [],
                   'stderrs' : [],
                   'lengths' : [],
                   'angles' : []}
        for i, dataRef  in enumerate(handles):
            trailed_glints = dataRef.get()
            visit = dataRef.dataId['visit']
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
