__author__ = 'Usai'
__date__ = '27.03.2026'

import string
from . import cwvector3d
import element_controller as ec


class CwLine3d:
    """Line segment in 3D defined by a start and end point as vectors."""

    def __init__(self, start: "cwvector3d.CwVector3d", end: "cwvector3d.CwVector3d"):
        """Args:
            start: First endpoint.
            end: Second endpoint.
        """
        self._start = start
        self._end = end

    @property
    def start(self) -> "cwvector3d.CwVector3d":
        """First endpoint."""
        return self._start

    @property
    def end(self) -> "cwvector3d.CwVector3d":
        """Second endpoint."""
        return self._end
    
    @property
    def length(self) -> float:
        """Length of the line."""
        return (self._end - self._start).magnitude()
    
    @property
    def midpoint(self) -> "cwvector3d.CwVector3d":
        """Midpoint of the line."""
        return (self._start + self._end) * 0.5

    def bake(self) -> int:
        """ add the line to the cadwork model """
        line_id = ec.create_line_points(self.start.point_3d, self.end.point_3d)
        return line_id

    def __str__(self) -> str:
        return f"Start={self._start}, End={self._end}"

    def __repr__(self) -> str:
        return f"CwLine3d(Start={self._start!r}, End={self._end!r})"

    def __eq__(self, other: "CwLine3d") -> bool:
        if not isinstance(other, CwLine3d):
            return NotImplemented
        return self._start == other._start and self._end == other._end

    def __ne__(self, other: "CwLine3d") -> bool:
        return not self.__eq__(other)
