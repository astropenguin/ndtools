# dependencies
import numpy as np
import ndtools.comparison.operators as op


def test_eq() -> None:
    left, right = np.arange(3), 1
    assert all(op.eq(left, right) == np.array([False, True, False]))


def test_eq_by_ne() -> None:
    left, right = np.arange(3), 1
    assert all(op.eq_by_ne(left, right) == np.array([False, True, False]))


def test_ge() -> None:
    left, right = np.arange(3), 1
    assert all(op.ge(left, right) == np.array([False, True, True]))


def test_ge_by_gt() -> None:
    left, right = np.arange(3), 1
    assert all(op.ge_by_gt(left, right) == np.array([False, True, True]))


def test_ge_by_le() -> None:
    left, right = np.arange(3), 1
    assert all(op.ge_by_le(left, right) == np.array([False, True, True]))


def test_ge_by_lt() -> None:
    left, right = np.arange(3), 1
    assert all(op.ge_by_lt(left, right) == np.array([False, True, True]))


def test_gt() -> None:
    left, right = np.arange(3), 1
    assert all(op.gt(left, right) == np.array([False, False, True]))


def test_gt_by_ge() -> None:
    left, right = np.arange(3), 1
    assert all(op.gt_by_ge(left, right) == np.array([False, False, True]))


def test_gt_by_le() -> None:
    left, right = np.arange(3), 1
    assert all(op.gt_by_le(left, right) == np.array([False, False, True]))


def test_gt_by_lt() -> None:
    left, right = np.arange(3), 1
    assert all(op.gt_by_lt(left, right) == np.array([False, False, True]))


def test_le() -> None:
    left, right = np.arange(3), 1
    assert all(op.le(left, right) == np.array([True, True, False]))


def test_le_by_ge() -> None:
    left, right = np.arange(3), 1
    assert all(op.le_by_ge(left, right) == np.array([True, True, False]))


def test_le_by_gt() -> None:
    left, right = np.arange(3), 1
    assert all(op.le_by_gt(left, right) == np.array([True, True, False]))


def test_le_by_lt() -> None:
    left, right = np.arange(3), 1
    assert all(op.le_by_lt(left, right) == np.array([True, True, False]))


def test_lt() -> None:
    left, right = np.arange(3), 1
    assert all(op.lt(left, right) == np.array([True, False, False]))


def test_lt_by_ge() -> None:
    left, right = np.arange(3), 1
    assert all(op.lt_by_ge(left, right) == np.array([True, False, False]))


def test_lt_by_gt() -> None:
    left, right = np.arange(3), 1
    assert all(op.lt_by_gt(left, right) == np.array([True, False, False]))


def test_lt_by_le() -> None:
    left, right = np.arange(3), 1
    assert all(op.lt_by_le(left, right) == np.array([True, False, False]))


def test_ne() -> None:
    left, right = np.arange(3), 1
    assert all(op.ne(left, right) == np.array([True, False, True]))


def test_ne_by_eq() -> None:
    left, right = np.arange(3), 1
    assert all(op.ne_by_eq(left, right) == np.array([True, False, True]))
