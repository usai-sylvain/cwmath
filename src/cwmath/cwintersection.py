from __future__ import annotations

__author__ = 'Usai'
__date__ = '27.03.2026'

from typing import TYPE_CHECKING

from . import cwline3d
from . import cwplane3d
from . import cwvector3d

if TYPE_CHECKING:
    from . import cwelement

# Max distance between closest points on two lines to treat as intersecting.
_TOL = 1e-6

# |n·direction| below this >> line parallel to plane; |n1×n2| below >> parallel planes.
_DENOM_EPS = 1e-10


class CwIntersection:
    """Static intersection helpers for 3D primitives."""

    @classmethod
    def line_plane(
        cls,
        line: 'cwline3d.CwLine3d',
        plane: 'cwplane3d.CwPlane3d',
    ) -> 'tuple[cwvector3d.CwVector3d | None, float | None]':
        """Intersection of the infinite line through ``line`` with ``plane``.

        Uses the segment endpoints only to define the line direction.

        Returns:
            ``(point, param)`` with ``point = line.start + param * (line.end - line.start)``,
            or ``(None, None)`` if the line is parallel to the plane (no unique point,
            including when the line lies in the plane).
        """
        # Plane: n·P + D = 0 (n = plane.z_axis, D = plane.constant_d). Line: P(t) = start + t*direction.
        start = line.start
        direction = line.end - line.start
        normal = plane.z_axis                       # not necessarily unit length; matches constant_d
        denom = normal.dot(direction)
        if abs(denom) < _DENOM_EPS:
            return None, None                       # line parallel to plane (or in-plane)
        numer = -(normal.dot(start) + plane.constant_d)
        param = numer / denom
        return start + direction * param, param

    @classmethod
    def plane_plane(
        cls,
        plane_a: 'cwplane3d.CwPlane3d',
        plane_b: 'cwplane3d.CwPlane3d',
    ) -> 'cwline3d.CwLine3d | None':
        """Intersection of two non-parallel planes as an infinite 3D line.

        Uses ``n·P + D = 0`` for each plane. Direction is ``n_a x n_b``.

        Returns:
            ``CwLine3d`` with ``start`` a point on both planes and ``end`` offset
            along the intersection direction, or ``None`` if planes are parallel.
        """
        n1 = plane_a.z_axis
        n2 = plane_b.z_axis
        d1 = plane_a.constant_d
        d2 = plane_b.constant_d
        w = n1.cross(n2)
        len2 = w.dot(w)
        if len2 < _DENOM_EPS * _DENOM_EPS:
            return None
        # Point on line: with n·P = -D, use (d1*n2 - d2*n1)×w / |w|², d_i = -D_i >> (D2*n1 - D1*n2)×w / |w|²
        vec = n1 * d2 - n2 * d1
        if vec.magnitude() < _DENOM_EPS:
            origin = plane_a.origin
            point = cwvector3d.CwVector3d(origin.x, origin.y, origin.z)
            dist_b = n2.dot(point) + d2
            if abs(dist_b) < _TOL:
                start = point
            else:
                for axis in (plane_a.x_axis, plane_a.y_axis):
                    den = n2.dot(axis)
                    if abs(den) > _DENOM_EPS:
                        start = point - axis * (dist_b / den)
                        break
                else:
                    return None
        else:
            start = vec.cross(w) / len2
        end = start + w
        return cwline3d.CwLine3d(start, end)

    @classmethod
    def line_line(
        cls,
        line_a: 'cwline3d.CwLine3d',
        line_b: 'cwline3d.CwLine3d',
    ) -> 'tuple[cwvector3d.CwVector3d | None, float | None, float | None]':
        """Intersection of two infinite 3D lines defined by segment endpoints.

        Lines are ``start + param * (end - start)`` for each segment.

        Returns:
            ``(point, param_a, param_b)`` where ``point`` is the ``CwVector3d``
            intersection, ``param_a`` is the parameter on ``line_a``, and
            ``param_b`` is the parameter on ``line_b``. If there is no unique
            intersection (skew, parallel distinct, or degenerate direction),
            returns ``(None, None, None)``.
        """
        # L_a(s) = p0 + s*u, L_b(t) = q0 + t*v; minimize |L_a - L_b|^2 → linear system in s, t.
        p0 = line_a.start
        p1 = line_a.end
        q0 = line_b.start
        q1 = line_b.end
        u = p1 - p0
        v = q1 - q0
        if u.magnitude() < 1e-10 or v.magnitude() < 1e-10:
            return None, None, None
        w0 = p0 - q0
        a = u.dot(u)
        b = u.dot(v)
        c = v.dot(v)
        d = u.dot(w0)
        e = v.dot(w0)
        denom = a * c - b * b                       # |u|^2 |v|^2 - (u·v)^2; zero if directions parallel
        if abs(denom) < 1e-10:
            return None, None, None
        param_a = (b * e - c * d) / denom
        param_b = (a * e - b * d) / denom
        point_on_a = p0 + u * param_a
        point_on_b = q0 + v * param_b
        # Skew lines: closest points differ; only return if they coincide within tolerance.
        if (point_on_a - point_on_b).magnitude() < _TOL:
            return point_on_a, param_a, param_b
        return None, None, None

    @classmethod
    def element_element(
        cls,
        element_a: cwelement.CwElement,
        element_b: cwelement.CwElement,
    ) -> cwelement.CwElementIntersection:
        """Intersect two element boxes (P1–P3 / XL–ZL oriented boxes).

        Returns:
            :class:`~cwelement.CwElementIntersection` with ``intersection_lines`` (each
            :class:`~cwline3d.CwLine3d` is an edge of the overlap boundary) and
            ``intersection_average_point`` (mean of segment midpoints, or ``None`` if empty).
        """
        # Local import keeps cwintersection importable without pulling cwelement at module load.
        from . import cwelement

        if not isinstance(element_a, cwelement.CwElement) or not isinstance(element_b, cwelement.CwElement):
            raise TypeError("element_element expects two CwElement instances")
        return element_a.intersect(element_b)  # Delegates to box–box clip + midpoint average.
