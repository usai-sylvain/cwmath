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

from .cwbox3d import CwBoxIntersection
from .cwbox3d import CwOrientedBox3d
from .cwbox3d import intersection_lines_between_boxes
from .cwline3d import CwLine3d
from .cwplane3d import CwPlane3d
from .cwvector3d import CwVector3d

# Same fields as :class:`CwBoxIntersection`; kept for callers that think in terms of elements.
CwElementIntersection = CwBoxIntersection


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
    def _box(self) -> CwOrientedBox3d:
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
        return CwOrientedBox3d(origin, xl, yl, zl, lx, ly, lz)

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
