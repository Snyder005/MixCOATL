# Why mix type definitions in the schema? E.g. "I" vs np.int32?
import numpy as np

import lsst.afw.table as afwTable
import lsst.geom as geom
from mixcoatl.streaks.line import Line2D, LineSegment2D


class StreakRecord(afwTable.BaseRecord):

    def getLine(self) -> Line2D:
        return Line2D(rho=self["line_rho"], theta=self["line_theta"])

    def getLineSegment(self) -> LineSegment2D:
        return LineSegment2D.from_center_length(
            line=self.getLine(),
            u_center=self["line_u_center"],
            length=self["line_length"],
        )

    def setLineSegment(self, segment: LineSegment2D) -> None:
        self["line_rho"] = segment.line.rho
        self["line_theta"] = segment.line.theta
        self["line_u_center"] = segment.interval.center
        self["line_length"] = segment.length

#    def getOrientedBBox(self):
#        ...


class StreakTable(afwTable.SourceTable):

    def __init__(self, schema):
        super().__init__(schema)

        #self.rhoKey = schema["line_rho"].asKey()

    @classmethod
    def make(cls, schema):
        ...

    @staticmethod
    def makeMinimalSchema():

        schema = afwTable.SourceTable.makeMinimalSchema()

        schema.addField("detector", type="I")

        schema.addField(
            "line_rho",
            type=np.float64,
            units="pixel",
        )

        schema.addField(
            "line_theta",
            type=geom.Angle,
        )

        schema.addField(
            "line_u_center",
            type=np.float64,
            units="pixel",
        )

        schema.addField(
            "line_length",
            type=np.float64,
            units="pixel",
        )

        return schema


class StreakCatalog(afwTable.SourceCatalog):
    """Can add: vectorized geometry helpers, export routines, ML conversion helpers"""

    def addNew(): # return a StreakRecord not SourceRecord
        ...


if __name__ == "__main__":
    schema = StreakTable.makeMinimalSchema()
    table = StreakTable.make(schema)
    catalog = StreakCatalog(table)
