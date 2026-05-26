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
            f"rho={seg.line.rho:.2f}, "
            f"theta={seg.line.theta.asDegrees():.2f} deg, "
            f"length={seg.length:.2f})"
        )

    def __getitem__(self, key):
        return self._record[key]

    def __setitem__(self, key, value):
        self._record[key] = value

    @property
    def record(self) -> afwTable.SourceRecord:
        return self._record

    @property
    def line(self) -> Line2D:
        return Line2D(rho=self["line_rho"], theta=self["line_theta"])

    @property
    def line_segment(self) -> LineSegment2D:
        return LineSegment2D.from_center_length(
            line=self.line,
            u_center=self["line_u_center"],
            length=self["line_length"],
        )

    @line_segment.setter
    def line_segment(self, segment: LineSegment2D) -> None:
        self["line_rho"] = segment.line.rho
        self["line_theta"] = segment.line.theta
        self["line_u_center"] = segment.interval.center
        self["line_length"] = segment.length

    @property
    def footprint(self) -> afwDetect.Footprint:
        return self.record.getFootprint()

    @footprint.setter
    def footprint(self, footprint: afwDetect.Footprint) -> None:
        self.record.setFootprint(footprint)

class StreakSchema:

    @staticmethod
    def makeMinimalSchema() -> afwTable.Schema:

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


if __name__ == "__main__":
    schema = StreakSchema.makeMinimalSchema()
    table = afwTable.SourceTable.make(schema)
    catalog = afwTable.SourceCatalog(table)
