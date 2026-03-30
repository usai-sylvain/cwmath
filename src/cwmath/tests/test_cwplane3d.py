import pytest
from _pytest.python_api import approx

import cadwork

from cwmath import cwvector3d
from cwmath import cwplane3d


@pytest.fixture
def plane_1():
    """Plane z = 2480 (normal (0, 0, -1)) from point and two in-plane vectors."""
    point = cwvector3d.CwVector3d(-42.500000, 47.500000, 2480.000000)
    u = cwvector3d.CwVector3d(0.0, 1.0, 0.0)
    v = cwvector3d.CwVector3d(1.0, 0.0, 0.0)
    return cwplane3d.CwPlane3d(point, u, v)


def test_call(plane_1):
    point = cadwork.point_3d(-42.500000, 47.500000, 0.000000)
    result = plane_1(point)
    assert result == approx(2480.0)


def test_str(plane_1):
    result = str(plane_1)
    assert result == "Origin=[-42.5, 47.5, 2480.0], XAxis=[0.0, 1.0, 0.0], YAxis=[1.0, 0.0, 0.0]"


def test_repr(plane_1):
    result = repr(plane_1)
    assert result == "CwPlane3d(Origin=[-42.5, 47.5, 2480.0], XAxis=[0.0, 1.0, 0.0], YAxis=[1.0, 0.0, 0.0])"


def test_eq(plane_1):
    point = cwvector3d.CwVector3d(-42.500000, 47.500000, 2480.000000)
    u = cwvector3d.CwVector3d(0.0, 1.0, 0.0)
    v = cwvector3d.CwVector3d(1.0, 0.0, 0.0)
    plane_2 = cwplane3d.CwPlane3d(point, u, v)
    assert plane_1 == plane_2


def test_ne(plane_1):
    point = cwvector3d.CwVector3d(-42.500000, 47.500000, 2480.000000)
    u = cwvector3d.CwVector3d(0.0, 1.0, 0.0)
    v = cwvector3d.CwVector3d(1.0, 0.0, 1.0)
    plane_2 = cwplane3d.CwPlane3d(point, u, v)
    assert plane_1 != plane_2


def test_origin_and_axes_stored(plane_1):
    """Plane stores origin and x_axis, y_axis; coefficients are derived."""
    assert plane_1.origin.x == approx(-42.5)
    assert plane_1.origin.y == approx(47.5)
    assert plane_1.origin.z == approx(2480.0)
    assert plane_1.x_axis == cwvector3d.CwVector3d(0.0, 1.0, 0.0)
    assert plane_1.y_axis == cwvector3d.CwVector3d(1.0, 0.0, 0.0)


def test_is_parallel(plane_1):
    point = cwvector3d.CwVector3d(-42.500000, 47.500000, 2480.000000)
    u = cwvector3d.CwVector3d(0.0, 1.0, 0.0)
    v = cwvector3d.CwVector3d(1.0, 0.0, 0.0)
    plane_2 = cwplane3d.CwPlane3d(point, u, v)
    assert plane_1.is_parallel(plane_2)


def test_is_perpendicular(plane_1):
    point = cwvector3d.CwVector3d(-42.500000, 47.500000, 2480.000000)
    u = cwvector3d.CwVector3d(0.0, 1.0, 0.0)
    v = cwvector3d.CwVector3d(0.0, 0.0, 1.0)
    plane_2 = cwplane3d.CwPlane3d(point, u, v)
    assert plane_1.is_perpendicular(plane_2)


def test_is_coplanar(plane_1):
    point = cwvector3d.CwVector3d(-42.500000, 47.500000, 280.000000)
    u = cwvector3d.CwVector3d(0.0, 1.0, 0.0)
    v = cwvector3d.CwVector3d(1.0, 0.0, 0.0)
    plane_2 = cwplane3d.CwPlane3d(point, u, v)
    assert plane_1.is_coplanar(plane_2)


def test_is_point_on_plane(plane_1):
    point = cwvector3d.CwVector3d(-42.500000, 47.500000, 2480.000000)
    assert plane_1.is_point_on_plane(point)


def test_distance_to_point(plane_1):
    point = cwvector3d.CwVector3d(-42.500000, 47.500000, 0.000000)
    result = plane_1.distance_to_point(point)
    assert result == approx(2480.0)


def test_distance_to_plane(plane_1):
    point = cwvector3d.CwVector3d(-42.500000, 47.500000, 2480.000000)
    u = cwvector3d.CwVector3d(0.0, 1.0, 0.0)
    v = cwvector3d.CwVector3d(1.0, 0.0, 0.0)
    plane2 = cwplane3d.CwPlane3d(point, u, v)
    result = plane_1.distance_to_plane(plane2)
    assert result == approx(0.0)
