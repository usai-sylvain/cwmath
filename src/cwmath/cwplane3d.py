__author__ = 'Brunner'
__date__ = '13.03.2024'

import cadwork
from cwmath import cwvector3d


class CwPlane3d:
    """Plane defined by an origin point and two orthonormal-style axes in the plane.

    Stored as origin, x_axis, y_axis. The implicit form Ax + By + Cz + D = 0
    is derived when needed: normal = x_axis * y_axis, then A,B,C from normal,
    D = -(A*origin.x + B*origin.y + C*origin.z).
    """

    def __init__(
        self,
        origin: "cadwork.point_3d",
        vector_x: "cwvector3d.CwVector3d",
        vector_y: "cwvector3d.CwVector3d",
    ):
        """Define the plane by one point and two direction vectors in the plane.

        The vectors are normalized and stored as x_axis and y_axis. The normal is
        x_axis * y_axis; A, B, C, D are computed from that when needed.

        Args:
            origin: A point on the plane (cadwork.point_3d).
            vector_x: First direction vector in the plane (not parallel to vector_y).
            vector_y: Second direction vector in the plane (not parallel to vector_x).
        """
        self._origin = origin
        self._x_axis = vector_x.normalize()
        self._y_axis = vector_y.normalize()

    @property
    def origin(self) -> cadwork.point_3d:
        """A point on the plane (cadwork.point_3d)."""
        return self._origin

    @property
    def x_axis(self) -> "cwvector3d.CwVector3d":
        """Unit direction vector in the plane (first axis)."""
        return self._x_axis

    @property
    def y_axis(self) -> "cwvector3d.CwVector3d":
        """Unit direction vector in the plane (second axis)."""
        return self._y_axis

    @property
    def z_axis(self) -> "cwvector3d.CwVector3d":
        """Normal vector of the plane (x_axis * y_axis)."""
        return self._x_axis.cross(self._y_axis)

    @property
    def _d(self) -> float:
        """Constant term in plane equation: -normal*origin."""
        normal = self.z_axis
        return -(normal.x * self._origin.x + normal.y * self._origin.y + normal.z * self._origin.z)

    def _signed_distance(self, point: cadwork.point_3d) -> float:
        """Signed distance from point to plane: normal*p + d."""
        normal = self.z_axis
        return normal.x * point.x + normal.y * point.y + normal.z * point.z + self._d

    @property
    def coefficient_a(self) -> float:
        """Coefficient A in Ax + By + Cz + D = 0 (read-only)."""
        return self.z_axis.x

    @property
    def coefficient_b(self) -> float:
        """Coefficient B in Ax + By + Cz + D = 0 (read-only)."""
        return self.z_axis.y

    @property
    def coefficient_c(self) -> float:
        """Coefficient C in Ax + By + Cz + D = 0 (read-only)."""
        return self.z_axis.z

    @property
    def constant_d(self) -> float:
        """Constant D in Ax + By + Cz + D = 0 (read-only)."""
        return self._d

    def __call__(self, point: cadwork.point_3d) -> float:
        """Value of the plane at point: normal*p + d (zero on the plane)."""
        return self._signed_distance(point)

    def __str__(self) -> str:
        return (
            f"Origin=[{self._origin.x}, {self._origin.y}, {self._origin.z}], "
            f"XAxis=[{self.x_axis.x}, {self.x_axis.y}, {self.x_axis.z}], "
            f"YAxis=[{self.y_axis.x}, {self.y_axis.y}, {self.y_axis.z}]"
        )

    def __repr__(self) -> str:
        return (
            f"CwPlane3d(Origin=[{self._origin.x}, {self._origin.y}, {self._origin.z}], "
            f"XAxis=[{self.x_axis.x}, {self.x_axis.y}, {self.x_axis.z}], "
            f"YAxis=[{self.y_axis.x}, {self.y_axis.y}, {self.y_axis.z}])"
        )

    def __eq__(self, other: "CwPlane3d") -> bool:
        if not isinstance(other, CwPlane3d):
            return NotImplemented
        normal = self.z_axis
        other_normal = other.z_axis
        return (
            abs(normal.x - other_normal.x) < 1e-6
            and abs(normal.y - other_normal.y) < 1e-6
            and abs(normal.z - other_normal.z) < 1e-6
            and abs(self._d - other._d) < 1e-6
        )

    def __ne__(self, other: "CwPlane3d") -> bool:
        return not self.__eq__(other)

    def is_parallel(self, other: "CwPlane3d") -> bool:
        """True if the two planes have parallel normals."""
        return self.z_axis.cross(other.z_axis).magnitude() < 1e-6

    def is_perpendicular(self, other: "CwPlane3d") -> bool:
        """True if the two planes have perpendicular normals."""
        return abs(self.z_axis.dot(other.z_axis)) < 1e-6

    def is_coplanar(self, other: "CwPlane3d") -> bool:
        """True if the two planes coincide (same normal and same d up to scale)."""
        normal = self.z_axis
        other_normal = other.z_axis
        if abs(other_normal.x) < 1e-10:
            return abs(normal.x) < 1e-10
        if abs(other_normal.y) < 1e-10:
            return abs(normal.y) < 1e-10
        if abs(other_normal.z) < 1e-10:
            return abs(normal.z) < 1e-10
        return (
            abs(normal.x / other_normal.x - normal.y / other_normal.y) < 1e-6
            and abs(normal.y / other_normal.y - normal.z / other_normal.z) < 1e-6
            and abs(self._d / other._d - normal.x / other_normal.x) < 1e-6
        )

    def is_point_on_plane(self, point: "cadwork.point_3d") -> bool:
        """True if the point lies on the plane."""
        return abs(self._signed_distance(point)) < 1e-6

    def distance_to_point(self, point: "cadwork.point_3d") -> float:
        """Distance from the point to the plane."""
        normal = self.z_axis
        return abs(self._signed_distance(point)) / normal.magnitude()

    def distance_to_plane(self, other: "CwPlane3d") -> float:
        """Distance between two parallel planes (undefined if not parallel)."""
        normal = self.z_axis
        return abs(self._d - other._d) / normal.magnitude()
