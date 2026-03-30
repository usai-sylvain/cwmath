import pytest
from _pytest.python_api import approx

from cwmath import cwintersection
from cwmath import cwvector3d
from cwmath.cwelement import _BoxGeom
from cwmath.cwelement import intersection_lines_between_boxes


def _axis_aligned() -> tuple[
    cwvector3d.CwVector3d,
    cwvector3d.CwVector3d,
    cwvector3d.CwVector3d,
]:
    return (
        cwvector3d.CwVector3d(1, 0, 0),
        cwvector3d.CwVector3d(0, 1, 0),
        cwvector3d.CwVector3d(0, 0, 1),
    )


def test_intersection_lines_disjoint_boxes():
    xl, yl, zl = _axis_aligned()
    a = _BoxGeom(cwvector3d.CwVector3d(0, 0, 0), xl, yl, zl, 1.0, 1.0, 1.0)
    b = _BoxGeom(cwvector3d.CwVector3d(5, 0, 0), xl, yl, zl, 1.0, 1.0, 1.0)
    got = intersection_lines_between_boxes(a, b)
    assert got.intersection_lines == []
    assert got.intersection_average_point is None


def test_intersection_lines_overlapping_cubes_twelve_edges():
    xl, yl, zl = _axis_aligned()
    a = _BoxGeom(cwvector3d.CwVector3d(0, 0, 0), xl, yl, zl, 2.0, 2.0, 2.0)
    b = _BoxGeom(cwvector3d.CwVector3d(1, 0, 0), xl, yl, zl, 2.0, 2.0, 2.0)
    got = intersection_lines_between_boxes(a, b)
    assert len(got.intersection_lines) == 12
    # Overlap [1,2] x [0,2] x [0,2]
    p = got.intersection_average_point
    assert p is not None
    assert p.x == approx(1.5)
    assert p.y == approx(1.0)
    assert p.z == approx(1.0)


def test_element_element_type_error():
    with pytest.raises(TypeError, match="CwElement"):
        cwintersection.CwIntersection.element_element(1, 2)
