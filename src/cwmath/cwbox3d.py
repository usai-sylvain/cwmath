"""Oriented box (OBB) geometry: corners, edges, planes, and box–box intersection segments.

This module is cadwork‑agnostic; :mod:`cwelement` builds :class:`CwOrientedBox3d` from API data.
"""

from __future__ import annotations

__author__ = "Usai"
__date__ = "30.03.2026"

from typing import NamedTuple

from .cwline3d import CwLine3d
from .cwplane3d import CwPlane3d
from .cwvector3d import CwVector3d

# Segment clipping / dedup (aligned with cwintersection tolerances order of magnitude).
_CLIP_TOL = 1e-6
_PARALLEL_EPS = 1e-12


class CwBoxIntersection(NamedTuple):
    """Result of intersecting two oriented boxes.

    ``intersection_average_point`` is the mean of each segment's midpoint, or ``None``
    when ``intersection_lines`` is empty.
    """

    intersection_lines: list[CwLine3d]
    intersection_average_point: CwVector3d | None


class CwOrientedBox3d(NamedTuple):
    """Oriented box: ``origin`` = min corner; orthonormal XL/YL/ZL; non‑negative Lx/Ly/Lz."""

    origin: CwVector3d
    xl: CwVector3d
    yl: CwVector3d
    zl: CwVector3d
    lx: float
    ly: float
    lz: float

    def corner(self, ix: int, iy: int, iz: int) -> CwVector3d:
        """Vertex of the box: indices 0/1 along local X/Y/Z (not cadwork P1–P3)."""
        return (
            self.origin
            + self.xl * (self.lx * ix)
            + self.yl * (self.ly * iy)
            + self.zl * (self.lz * iz)
        )

    def line(self, a: tuple[int, int, int], b: tuple[int, int, int]) -> CwLine3d:
        return CwLine3d(self.corner(*a), self.corner(*b))

    def plane(self, c0: tuple[int, int, int], c1: tuple[int, int, int], c2: tuple[int, int, int]) -> CwPlane3d:
        o = self.corner(*c0)
        u = self.corner(*c1) - o
        v = self.corner(*c2) - o
        return CwPlane3d(o.point_3d, u, v)

    def iter_edges(self) -> tuple[CwLine3d, ...]:
        """The twelve edges of the box."""
        b = self.line
        return (
            # Bottom face: two edges parallel to Y (iz=0; ix=0 and ix=1).
            b((0, 0, 0), (0, 1, 0)),
            b((1, 0, 0), (1, 1, 0)),
            # Top face: two edges parallel to Y (iz=1).
            b((0, 0, 1), (0, 1, 1)),
            b((1, 0, 1), (1, 1, 1)),
            # Bottom / top: four edges parallel to X (vary iy, iz).
            b((0, 0, 0), (1, 0, 0)),
            b((0, 1, 0), (1, 1, 0)),
            b((0, 0, 1), (1, 0, 1)),
            b((0, 1, 1), (1, 1, 1)),
            # Four vertical edges parallel to Z (back-left, back-right, front-left, front-right).
            b((0, 0, 0), (0, 0, 1)),
            b((0, 1, 0), (0, 1, 1)),
            b((1, 0, 0), (1, 0, 1)),
            b((1, 1, 0), (1, 1, 1)),
        )


def _clip_segment_to_box(line: CwLine3d, box: CwOrientedBox3d) -> CwLine3d | None:
    """Clip ``line`` to the closed box; ``None`` if disjoint or degenerate."""
    o = box.origin
    start = line.start - o  # Box-local frame; origin = min corner (see :class:`CwOrientedBox3d`).
    d = line.end - line.start
    t_lo, t_hi = 0.0, 1.0  # Keep only the portion of the segment with t in [0, 1].
    axes = (box.xl, box.yl, box.zl)
    limits = (box.lx, box.ly, box.lz)

    # Slab test per axis: 0 <= axis·(start + t*d) <= lim  =>  intersect [t_lo, t_hi] with valid t range.
    for axis, lim in zip(axes, limits):
        v0 = axis.dot(start)
        vd = axis.dot(d)
        if abs(vd) < _PARALLEL_EPS:
            # Segment parallel to this pair of faces; must lie between them.
            if v0 < -_CLIP_TOL or v0 > lim + _CLIP_TOL:
                return None
            continue
        t_a = (0.0 - v0) / vd
        t_b = (lim - v0) / vd
        if t_a > t_b:
            t_a, t_b = t_b, t_a
        t_lo = max(t_lo, t_a)
        t_hi = min(t_hi, t_b)
        if t_lo > t_hi + _CLIP_TOL:
            return None

    p0 = line.start + d * t_lo
    p1 = line.start + d * t_hi
    if (p1 - p0).magnitude() < _CLIP_TOL:
        return None  # Point-like overlap: treat as no line for downstream averaging.
    return CwLine3d(p0, p1)


def _line_segment_key(line: CwLine3d) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    # Rounded endpoints so numerically duplicate segments get one key; order-independent (a,b) vs (b,a).
    a = (round(line.start.x, 9), round(line.start.y, 9), round(line.start.z, 9))
    b = (round(line.end.x, 9), round(line.end.y, 9), round(line.end.z, 9))
    return (a, b) if a <= b else (b, a)


def _average_segment_midpoints(lines: list[CwLine3d]) -> CwVector3d | None:
    if not lines:
        return None
    total = CwVector3d(0.0, 0.0, 0.0)
    for ln in lines:
        total = total + (ln.start + ln.end) * 0.5  # Midpoint of each intersection segment.
    inv = 1.0 / len(lines)
    return total * inv  # Simple mean (not geometric centroid of the polyhedron).


def intersection_lines_between_boxes(a: CwOrientedBox3d, b: CwOrientedBox3d) -> CwBoxIntersection:
    """Intersect two oriented boxes.

    Each edge of either box is clipped to the other; segment midpoints are averaged
    into :py:attr:`CwBoxIntersection.intersection_average_point`.
    """
    seen: set[tuple[tuple[float, float, float], tuple[float, float, float]]] = set()
    out: list[CwLine3d] = []
    # Edges of A clipped to the inside of B yield boundary segments of A ∩ B.
    for edge in a.iter_edges():
        clipped = _clip_segment_to_box(edge, b)
        if clipped is None:
            continue
        key = _line_segment_key(clipped)
        if key in seen:
            continue  # Same segment can appear when clipping B's edges later.
        seen.add(key)
        out.append(clipped)
    # Same using B's edges against A (covers segments only contributed from B).
    for edge in b.iter_edges():
        clipped = _clip_segment_to_box(edge, a)
        if clipped is None:
            continue
        key = _line_segment_key(clipped)
        if key in seen:
            continue
        seen.add(key)
        out.append(clipped)
    return CwBoxIntersection(out, _average_segment_midpoints(out))
