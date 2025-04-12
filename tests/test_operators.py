# dependencies
import numpy as np
import ndtools.operators as op


def test_eq_by_ne() -> None:
    left, right = np.arange(3), 1
    expected = np.array([False, True, False])
    assert (op.eq_by_ne(left, right) == expected).all()


def test_ge_by_gt() -> None:
    left, right = np.arange(3), 1
    expected = np.array([False, True, True])
    assert (op.ge_by_gt(left, right) == expected).all()


def test_ge_by_le() -> None:
    left, right = np.arange(3), 1
    expected = np.array([False, True, True])
    assert (op.ge_by_le(left, right) == expected).all()


def test_ge_by_lt() -> None:
    left, right = np.arange(3), 1
    expected = np.array([False, True, True])
    assert (op.ge_by_lt(left, right) == expected).all()


def test_gt_by_ge() -> None:
    left, right = np.arange(3), 1
    expected = np.array([False, False, True])
    assert (op.gt_by_ge(left, right) == expected).all()


def test_gt_by_le() -> None:
    left, right = np.arange(3), 1
    expected = np.array([False, False, True])
    assert (op.gt_by_le(left, right) == expected).all()


def test_gt_by_lt() -> None:
    left, right = np.arange(3), 1
    expected = np.array([False, False, True])
    assert (op.gt_by_lt(left, right) == expected).all()


def test_le_by_ge() -> None:
    left, right = np.arange(3), 1
    expected = np.array([True, True, False])
    assert (op.le_by_ge(left, right) == expected).all()


def test_le_by_gt() -> None:
    left, right = np.arange(3), 1
    expected = np.array([True, True, False])
    assert (op.le_by_gt(left, right) == expected).all()


def test_le_by_lt() -> None:
    left, right = np.arange(3), 1
    expected = np.array([True, True, False])
    assert (op.le_by_lt(left, right) == expected).all()


def test_lt_by_ge() -> None:
    left, right = np.arange(3), 1
    expected = np.array([True, False, False])
    assert (op.lt_by_ge(left, right) == expected).all()


def test_lt_by_gt() -> None:
    left, right = np.arange(3), 1
    expected = np.array([True, False, False])
    assert (op.lt_by_gt(left, right) == expected).all()


def test_lt_by_le() -> None:
    left, right = np.arange(3), 1
    expected = np.array([True, False, False])
    assert (op.lt_by_le(left, right) == expected).all()


def test_ne_by_eq() -> None:
    left, right = np.arange(3), 1
    expected = np.array([True, False, True])
    assert (op.ne_by_eq(left, right) == expected).all()
