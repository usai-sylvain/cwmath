"""Element geometry helpers.

**Cadwork geometry** (axis / references, not box vertices):

- **``P1``** — start of the element **axis**; it lies at the **centre of the cross‑section** on the
  **back** face of the box (local ``ix = 0``), not at a corner.
- **``P2``** — separated from ``P1`` along **+XL** (length direction). **Lx** comes from
  ``geometry_controller.get_length`` (along that axis).
- **``P3``** — a short step from ``P1`` along local **Y** (e.g. ~1 mm) to fix **YL**; it does **not**
  define **Ly** / **Lz**.

**Section size** uses ``get_width`` → **Ly** (local **Y**) and ``get_height`` → **Lz** (local **Z**).

The **min corner** (back–left–bottom) of the OBB is::

  ``origin = P1 - (Ly/2)·YL - (Lz/2)·ZL``

so ``P1`` is the midpoint of the back face. Vertices are ``origin + ix·Lx·XL + iy·Ly·YL +
iz·Lz·ZL`` for ``ix, iy, iz`` ∈ {0,1}.

Face names (*back*, *front*, …) refer to that derived box and its local indices.
"""

from __future__ import annotations

__author__ = "Usai"
__date__ = "27.03.2026"

from functools import cached_property
from typing import NamedTuple

from .cwline3d import CwLine3d
from .cwplane3d import CwPlane3d
from .cwvector3d import CwVector3d

# Segment clipping / dedup (aligned with cwintersection tolerances order of magnitude).
_CLIP_TOL = 1e-6
_PARALLEL_EPS = 1e-12


class CwElementIntersection(NamedTuple):
    """Result of intersecting two element boxes.

    ``intersection_average_point`` is the mean of each segment's midpoint, or ``None``
    when ``intersection_lines`` is empty.
    """

    intersection_lines: list[CwLine3d]
    intersection_average_point: CwVector3d | None


class _BoxGeom(NamedTuple):
    """Oriented box: ``origin`` = min corner; orthonormal XL/YL/ZL; non‑negative Lx/Ly/Lz."""

    origin: CwVector3d
    xl: CwVector3d
    yl: CwVector3d
    zl: CwVector3d
    lx: float
    ly: float
    lz: float

    def corner(self, ix: int, iy: int, iz: int) -> CwVector3d:
        """Vertex of the derived box: indices 0/1 along local X/Y/Z (not cadwork P1–P3)."""
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


def _clip_segment_to_box(line: CwLine3d, box: _BoxGeom) -> CwLine3d | None:
    """Clip ``line`` to the closed box; ``None`` if disjoint or degenerate."""
    o = box.origin
    start = line.start - o  # Box-local frame; origin = min corner (see :class:`_BoxGeom`).
    d = line.end - line.start
    t_lo, t_hi = 0.0, 1.0  # Keep only the portion of the segment with t in [0, 1].
    axes = (box.xl, box.yl, box.zl)
    limits = (box.lx, box.ly, box.lz)

    # Slab test per axis: 0 <= axis·(start + t*d) <= lim  =>  intersect [t_lo, t_hi] with valid t range.
    for axis, lim in zip(axes, limits):
        v0 = axis.dot(start)  # Projection of segment start onto this axis.
        vd = axis.dot(d)  # Rate of change of that projection along the segment.
        if abs(vd) < _PARALLEL_EPS:
            # Segment parallel to this pair of faces; must lie between them.
            if v0 < -_CLIP_TOL or v0 > lim + _CLIP_TOL:
                return None
            continue
        t_a = (0.0 - v0) / vd  # t where projection hits the "0" face.
        t_b = (lim - v0) / vd  # t where projection hits the "lim" face.
        if t_a > t_b:
            t_a, t_b = t_b, t_a
        t_lo = max(t_lo, t_a)
        t_hi = min(t_hi, t_b)
        if t_lo > t_hi + _CLIP_TOL:
            return None  # Empty intersection along this axis => segment misses the box.

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


def intersection_lines_between_boxes(a: _BoxGeom, b: _BoxGeom) -> CwElementIntersection:
    """Intersect two oriented boxes.

    Each edge of either box is clipped to the other; segment midpoints are averaged
    into :py:attr:`CwElementIntersection.intersection_average_point`.
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
    return CwElementIntersection(out, _average_segment_midpoints(out))


class CwElement:
    """CAD element by id; OBB from axis ``P1`` (back-face section centre), ``get_length`` / width / height, and local axes."""

    def __init__(self, id: int):
        """Args:
            id: Element identifier.
        """
        self._id = id

    @property
    def id(self) -> int:
        """Element identifier."""
        return self._id

    @cached_property
    def _box(self) -> _BoxGeom:
        import geometry_controller as gc

        p1 = CwVector3d.from_point_3d(gc.get_p1(self._id))
        xl = CwVector3d.from_point_3d(gc.get_xl(self._id)).normalize()
        yl = CwVector3d.from_point_3d(gc.get_yl(self._id)).normalize()
        zl = CwVector3d.from_point_3d(gc.get_zl(self._id)).normalize()
        lx = float(gc.get_length(self._id))
        ly = float(gc.get_width(self._id))
        lz = float(gc.get_height(self._id))
        # P1: axis start = centre of back face (ix=0); min corner is half extents in -Y and -Z.
        origin = p1 - yl * (ly * 0.5) - zl * (lz * 0.5)
        return _BoxGeom(origin, xl, yl, zl, lx, ly, lz)

    # --- Edges (12) ------------------------------------------------------------
    @property
    def edges(self) -> list[CwLine3d]:
        """All edges of the element."""
        return list(self._box.iter_edges())

    @property
    def bottom_back_line(self) -> CwLine3d:
        """Bottom edge on the back face, left -> right."""
        return self._box.line((0, 0, 0), (0, 1, 0))

    @property
    def bottom_front_line(self) -> CwLine3d:
        """Bottom edge on the front face, left -> right."""
        return self._box.line((1, 0, 0), (1, 1, 0))

    @property
    def top_back_line(self) -> CwLine3d:
        """Top edge on the back face, left -> right."""
        return self._box.line((0, 0, 1), (0, 1, 1))

    @property
    def top_front_line(self) -> CwLine3d:
        """Top edge on the front face, left -> right."""
        return self._box.line((1, 0, 1), (1, 1, 1))

    @property
    def bottom_left_line(self) -> CwLine3d:
        """Bottom edge on the left side, back -> front."""
        return self._box.line((0, 0, 0), (1, 0, 0))

    @property
    def bottom_right_line(self) -> CwLine3d:
        """Bottom edge on the right side, back -> front."""
        return self._box.line((0, 1, 0), (1, 1, 0))

    @property
    def top_left_line(self) -> CwLine3d:
        """Top edge on the left side, back -> front."""
        return self._box.line((0, 0, 1), (1, 0, 1))

    @property
    def top_right_line(self) -> CwLine3d:
        """Top edge on the right side, back -> front."""
        return self._box.line((0, 1, 1), (1, 1, 1))

    @property
    def back_left_line(self) -> CwLine3d:
        """Back edge on the left side, bottom -> top."""
        return self._box.line((0, 0, 0), (0, 0, 1))

    @property
    def back_right_line(self) -> CwLine3d:
        """Back edge on the right side, bottom -> top."""
        return self._box.line((0, 1, 0), (0, 1, 1))

    @property
    def front_left_line(self) -> CwLine3d:
        """Front edge on the left side, bottom -> top."""
        return self._box.line((1, 0, 0), (1, 0, 1))

    @property
    def front_right_line(self) -> CwLine3d:
        """Front edge on the right side, bottom -> top."""
        return self._box.line((1, 1, 0), (1, 1, 1))

    # --- Faces (6) -------------------------------------------------------------

    @property
    def bottom_plane(self) -> CwPlane3d:
        """Local ``iz=0`` face through the box min corner; spanned by XL and YL (see module doc)."""
        b = self._box
        return CwPlane3d(b.origin.point_3d, b.xl, b.yl)

    @property
    def top_plane(self) -> CwPlane3d:
        """Top face."""
        return self._box.plane((0, 0, 1), (1, 0, 1), (0, 1, 1))

    @property
    def back_plane(self) -> CwPlane3d:
        """Back face (x = 0 in local corner indices)."""
        return self._box.plane((0, 0, 0), (0, 1, 0), (0, 0, 1))

    @property
    def front_plane(self) -> CwPlane3d:
        """Front face."""
        return self._box.plane((1, 0, 0), (1, 1, 0), (1, 0, 1))

    @property
    def left_plane(self) -> CwPlane3d:
        """Left face (y = 0)."""
        return self._box.plane((0, 0, 0), (1, 0, 0), (0, 0, 1))

    @property
    def right_plane(self) -> CwPlane3d:
        """Right face."""
        return self._box.plane((0, 1, 0), (1, 1, 0), (0, 1, 1))

    @property
    def plane(self) -> CwPlane3d:
        """Same as :py:attr:`bottom_plane` (local bottom face of the derived box)."""
        return self.bottom_plane

    def intersect(self, other: CwElement) -> CwElementIntersection:
        """Intersect this element's box with ``other``'s box.

        Returns intersection segments and the average of their midpoints.
        """
        # Uses cached orthonormal boxes; geometry_controller runs on first _box access per element.
        return intersection_lines_between_boxes(self._box, other._box)

    def __str__(self) -> str:
        return f"id={self._id}"

    def __repr__(self) -> str:
        return f"CwElement(id={self._id!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CwElement):
            return NotImplemented
        return self._id == other._id

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)
