import numpy as np

import lsst.afw.table as afwTable
import lsst.geom as geom


def makeStreakSchema() -> afwTable.Schema:

    schema = afwTable.Schema()
    
    schema.addField(
        "detector",
        type="I",
        doc="Detector ID",
    )    
    schema.addField(
        "line_rho",
        type=np.float64,
        units="pixel",
        doc="Streak perpendicular distance from origin in pixel coordinates",
    )
    schema.addField(
        "line_theta",
        type=geom.Angle,
        doc="Streak angle from origin in pixel coordinates"
    )

    return schema


def makeStreakCatalog(schema: afwTable.Schema | None = None) -> afwTable.BaseCatalog:
    
    if schema is None:
        schema = makeStreakSchema()

    return afwTable.BaseCatalog(schema)
