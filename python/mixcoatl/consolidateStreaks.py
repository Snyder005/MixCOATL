import numpy as np

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.daf.base as dafBase
import lsst.pipe.base.connectionTypes as cT

class ConsolidateStreaksConnections(pipeBase.PipelineTaskConnections, 
                                    dimensions=("instrument", "visit")):
    streaks = cT.Input(
        doc="Streaks info.",
        name="streaks",
        storageClass="ArrowNumpyDict",
        dimensions=("instrument", "visit", "detector"),
        deferLoad=True,
        multiple=True,
    )
    streaks_summary = cT.Output(
        doc="Consolidated streaks info.",
        name="streaks_summary",
        storageClass="ArrowNumpyDict",
        dimensions=("instrument", "visit"),
    )

class ConsolidateStreaksConfig(pipeBase.PipelineTaskConfig,
                               pipelineConnections=ConsolidateStreaksConnections): pass

class ConsolidateStreaksTask(pipeBase.PipelineTask):
    _DefaultName = "consolidateStreaks"
    ConfigClass = ConsolidateStreaksConfig

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        handles = butlerQC.get(inputRefs.streaks)
        visit = handles[0].dataId['visit']

        results = self.run(visit=visit, handles=handles)

        butlerQC.put(results, outputRefs)

    def run(self, handles):
        
        summary = {
            'visit' : [],
            'detector' : [],
            'rho' : [],
            'theta' : [],
            'sigma' : [],
            'reducedChi2' : [],
            'modelMaximum' : []
        }
        for i, dataRef  in enumerate(handles):
            streaks = dataRef.get()
            detector = dataRef.dataId['detector']
            num_streaks = len(streaks['rho'])
            for j in range(num_streaks):
                summary['visit'].append(visit)
                summary['detector'].append(detector)
                for k in streaks:
                    summary[k].append(streaks[k][j])

        for k in summary:
            summary[k] = np.asarray(summary[k])

        result = pipeBase.Struct(streaks_summary=summary)
        return result
