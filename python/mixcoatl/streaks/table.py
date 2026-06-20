# Why mix type definitions in the schema? E.g. "I" vs np.int32?
# Optional Upgrades:
#   * Convenience geometric properties (center, length, p0, p1): 
#     These may be useful for downstream analysis code/ML labeling
import numpy as np

import lsst.afw.detection as afwDetect
import lsst.afw.table as afwTable
import lsst.geom as geom
from mixcoatl.streaks.line import Line2D, LineSegment2D


class StreakAdapter:

    def __init__(self, record: afwTable.SourceRecord):
        self._record = record

    def __repr__(self):
        seg = self.line_segment

        return (
            f"StreakAdapter("
            f"rho={seg.rho:.2f}, "
            f"theta={seg.theta.asDegrees():.2f} deg, "
            f"length={seg.length:.2f})"
        )

    def __getitem__(self, key):
        return self._record[key]

    def __setitem__(self, key, value):
        self._record[key] = value

    @property
    def record(self) -> afwTable.SourceRecord:
        return self._record

    def getLine(self) -> Line2D:
        return Line2D(rho=self["line_rho"], theta=self["line_theta"])

    def getLineSegment(self) -> LineSegment2D:
        return LineSegment2D.from_center_length(
            line=self.line,
            u_center=self["line_u_center"],
            length=self["line_length"],
        )

    def setLineSegment(self, segment: LineSegment2D) -> None:
        self["line_rho"] = segment.rho
        self["line_theta"] = segment.theta
        self["line_u_center"] = segment.interval.center
        self["line_length"] = segment.length

    def getFootprint(self) -> afwDetect.Footprint:
        return self.record.getFootprint()

    def setFootprint(self, footprint: afwDetect.Footprint) -> None:
        self.record.setFootprint(footprint)

    def getCentroid(self) -> geom.Point2D:
        return self.record.getCentroid()

    def getCoord(self) -> geom.SpherePoint:
        return self.record.getCoord()

    def setCoord(self, coord: geom.SpherePoint) -> None:
        self.record.setCoord(coord)

    @staticmethod
    def makeMinimalSchema() -> afwTable.Schema:
        schema = afwTable.SourceTable.makeMinimalSchema()
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
        centroidKey = afwTable.Point2DKey.addFields(
            schema,
            "line_center",
            "Line segment center",
            "pixel",
        )

        return schema


if __name__ == "__main__":
    schema = StreakAdapter.makeMinimalSchema()
    table = afwTable.SourceTable.make(schema)
    catalog = afwTable.SourceCatalog(table)
