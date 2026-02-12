# CwPlane3d

A plane defined by **origin** (a point) and **x_axis**, **y_axis** (two direction vectors in the plane). The normal is `x_axis * y_axis`; the implicit form `Ax + By + Cz + D = 0` is derived when needed.

**Constructor:** `CwPlane3d(origin, vector_x, vector_y)`

- `origin`: A point on the plane (`cadwork.point_3d`).
- `vector_x`, `vector_y`: Two direction vectors in the plane (`CwVector3d`, must not be parallel). Stored as `x_axis` and `y_axis` (normalized).

::: src.cwmath.cwplane3d
    options:
        show_root_heading: true
        show_source: true
