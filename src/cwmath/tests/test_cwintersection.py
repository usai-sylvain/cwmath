from _pytest.python_api import approx

from cwmath import cwintersection
from cwmath import cwline3d
from cwmath import cwplane3d
from cwmath import cwvector3d


def test_line_plane_hits_xy_plane():
    plane = cwplane3d.CwPlane3d(
        cwvector3d.CwVector3d(0, 0, 0),
        cwvector3d.CwVector3d(1, 0, 0),
        cwvector3d.CwVector3d(0, 1, 0),
    )
    line = cwline3d.CwLine3d(
        cwvector3d.CwVector3d(1, 1, 5),
        cwvector3d.CwVector3d(1, 1, -5),
    )
    p, param = cwintersection.CwIntersection.line_plane(line, plane)
    assert p is not None and param is not None
    assert p.x == approx(1.0)
    assert p.y == approx(1.0)
    assert p.z == approx(0.0)
    assert param == approx(0.5)


def test_line_plane_parallel_returns_none():
    plane = cwplane3d.CwPlane3d(
        cwvector3d.CwVector3d(0, 0, 0),
        cwvector3d.CwVector3d(1, 0, 0),
        cwvector3d.CwVector3d(0, 1, 0),
    )
    line = cwline3d.CwLine3d(
        cwvector3d.CwVector3d(0, 0, 1),
        cwvector3d.CwVector3d(1, 0, 1),
    )
    assert cwintersection.CwIntersection.line_plane(line, plane) == (None, None)


def test_line_line_intersect():
    a = cwline3d.CwLine3d(
        cwvector3d.CwVector3d(0, 0, 0),
        cwvector3d.CwVector3d(2, 0, 0),
    )
    b = cwline3d.CwLine3d(
        cwvector3d.CwVector3d(1, -1, 0),
        cwvector3d.CwVector3d(1, 1, 0),
    )
    p, param_a, param_b = cwintersection.CwIntersection.line_line(a, b)
    assert p is not None and param_a is not None and param_b is not None
    assert p.x == approx(1.0)
    assert p.y == approx(0.0)
    assert p.z == approx(0.0)
    assert param_a == approx(0.5)
    assert param_b == approx(0.5)


def test_line_line_skew_returns_none():
    a = cwline3d.CwLine3d(
        cwvector3d.CwVector3d(0, 0, 0),
        cwvector3d.CwVector3d(1, 0, 0),
    )
    b = cwline3d.CwLine3d(
        cwvector3d.CwVector3d(0, 1, 0),
        cwvector3d.CwVector3d(0, 1, 1),
    )
    assert cwintersection.CwIntersection.line_line(a, b) == (None, None, None)


def test_line_line_parallel_returns_none():
    a = cwline3d.CwLine3d(
        cwvector3d.CwVector3d(0, 0, 0),
        cwvector3d.CwVector3d(1, 0, 0),
    )
    b = cwline3d.CwLine3d(
        cwvector3d.CwVector3d(0, 1, 0),
        cwvector3d.CwVector3d(1, 1, 0),
    )
    assert cwintersection.CwIntersection.line_line(a, b) == (None, None, None)


def test_plane_plane_xy_and_xz_through_origin():
    xy = cwplane3d.CwPlane3d(
        cwvector3d.CwVector3d(0, 0, 0),
        cwvector3d.CwVector3d(1, 0, 0),
        cwvector3d.CwVector3d(0, 1, 0),
    )
    xz = cwplane3d.CwPlane3d(
        cwvector3d.CwVector3d(0, 0, 0),
        cwvector3d.CwVector3d(1, 0, 0),
        cwvector3d.CwVector3d(0, 0, 1),
    )
    ln = cwintersection.CwIntersection.plane_plane(xy, xz)
    assert ln is not None
    assert xy(ln.start.point_3d) == approx(0.0)
    assert xy(ln.end.point_3d) == approx(0.0)
    assert xz(ln.start.point_3d) == approx(0.0)
    assert xz(ln.end.point_3d) == approx(0.0)
    direction = ln.end - ln.start
    assert abs(direction.y) < 1e-9 and abs(direction.z) < 1e-9
    assert abs(direction.x) > 1e-9


def test_plane_plane_z2_and_y0():
    z2 = cwplane3d.CwPlane3d(
        cwvector3d.CwVector3d(0, 0, 2),
        cwvector3d.CwVector3d(1, 0, 0),
        cwvector3d.CwVector3d(0, 1, 0),
    )
    y0 = cwplane3d.CwPlane3d(
        cwvector3d.CwVector3d(0, 0, 0),
        cwvector3d.CwVector3d(1, 0, 0),
        cwvector3d.CwVector3d(0, 0, 1),
    )
    ln = cwintersection.CwIntersection.plane_plane(z2, y0)
    assert ln is not None
    for p in (ln.start, ln.end):
        assert p.y == approx(0.0)
        assert p.z == approx(2.0)


def test_plane_plane_parallel_returns_none():
    p0 = cwvector3d.CwVector3d(0, 0, 0)
    u = cwvector3d.CwVector3d(1, 0, 0)
    v = cwvector3d.CwVector3d(0, 1, 0)
    horizontal_z0 = cwplane3d.CwPlane3d(p0, u, v)
    horizontal_z1 = cwplane3d.CwPlane3d(cwvector3d.CwVector3d(0, 0, 1), u, v)
    assert cwintersection.CwIntersection.plane_plane(horizontal_z0, horizontal_z1) is None
