import math
import numpy as np
from astropy.table import Column
from sklearn.cluster import AgglomerativeClustering

import lsst.geom as geom
from lsst.afw.cameraGeom import PIXELS, FOCAL_PLANE
from lsst.pex.config import Field
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base.connectionTypes import Input, Output

from .utils import regularize_line, embed_line

__all__ = ["ClusterStreaksTask", "ClusterStreaksConfig"]

def line_to_focal_plane(rho, theta, detector):
    """Convert an LSST `Line` parameters to focal plane coordinate system."""
    theta = math.radians(theta)
    normal = geom.Extent2D([math.cos(theta), math.sin(theta)])
    direction = geom.Extent2D([-math.sin(theta), math.cos(theta)])
    point0 = detector.getBBox().getCenter() + normal * rho
    point1 = point0 + direction

    point0_fp = detector.transform(point0, PIXELS, FOCAL_PLANE)
    point1_fp = detector.transform(point1, PIXELS, FOCAL_PLANE)
    direction_fp = point0_fp - point1_fp
    direction_fp /= direction_fp.computeNorm()
    normal_fp = geom.Extent2D(-direction_fp.getY(), direction_fp.getX())

    rho_fp = normal_fp.getX() * point0_fp.getX() + normal_fp.getY() * point0_fp.getY()
    theta_fp = math.atan2(normal_fp.getY(), normal_fp.getX())

    return regularize_line(rho_fp, theta_fp)


class ClusterStreaksConnections(PipelineTaskConnections, dimensions=("instrument", "visit")):
    camera = Input(
        doc="Input camera for geometry information.",
        name="camera",
        storageClass="Camera",
        dimensions=["instrument"],
        isCalibration=True,
    )
    
    preliminary_streaks_visit = Input(
        doc="Input preliminary per-visit streaks info.",
        name="preliminary_streaks_visit",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "visit"),
    )
    streaks_visit = Output(
        doc="Finalized per-visit streaks info.",
        name="streaks_visit",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "visit"),
    )


class ClusterStreaksConfig(PipelineTaskConfig, pipelineConnections=ClusterStreaksConnections):
    rho_tol = Field(
        doc="Maximum tolerance for the difference in perpendicular distance, in mm.", 
        dtype=float,
        default=0.5,
    )
    theta_tol = Field(
        doc="Maximum tolerance for the difference in angle, in degrees.",
        dtype=float,
        default=0.5,
    )
    threshold = Field(
        doc="Parameter clustering distance threshold.",
        dtype=float,
        default=1.0,
    )


class ClusterStreaksTask(PipelineTask):
    """Add clustering labels to consolidated per-visit streaks table."""
    _DefaultName = "clusterStreaks"
    ConfigClass = ClusterStreaksConfig

    def run(self, preliminary_streaks_visit, camera):
        """Cluster streaks by Hesse normal parameterization.
        
        Parameters
        ----------
        preliminary_streaks_visit : `astropy.Table`
            Table of per-visit streaks.
        camera : `lsst.afw.cameraGeom.Camera`
            Camera to use for camera geometry information.
        
        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The results struct containing:
                
            ``streaks_visit``
                A table of per-visit streaks including clustering labels
                (`astropy.Table`).
        """ 
        points3d = []

        if len(preliminary_streaks_visit) > 0:
            for row in preliminary_streaks_visit:
                detector_id = int(row["detector"])
                detector = camera[detector_id]
                rho_fp, theta_fp = line_to_focal_plane(row["rho"], row["theta"], detector)

                rho_tol = self.config.rho_tol
                theta_tol = math.radians(self.config.theta_tol)
                points3d.append(embed_line(rho_fp, theta_fp, rho_tol, theta_tol))
            
            clustering = AgglomerativeClustering(
                n_clusters=None,
                linkage="average",
                distance_threshold=self.config.threshold,
            )

            labels = clustering.fit_predict(np.array(points3d))
        else:
            labels = []
            
        label_col = Column(name="cluster_label", data=labels, dtype="int32")
        preliminary_streaks_visit.add_column(label_col)
        
        # Patch fix to try to set detector column to int
        preliminary_streaks_visit["detector"] = preliminary_streaks_visit["detector"].astype("int32")

        return Struct(streaks_visit=preliminary_streaks_visit)
