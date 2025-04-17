# dependencies
import numpy as np
import ndtools.operators as op


def test_eq_by_ne() -> None:
    left, right = np.arange(3), 1
    assert (op.eq_by_ne(left, right) == np.array([False, True, False])).all()


def test_ge_by_gt() -> None:
    left, right = np.arange(3), 1
    assert (op.ge_by_gt(left, right) == np.array([False, True, True])).all()


def test_ge_by_le() -> None:
    left, right = np.arange(3), 1
    assert (op.ge_by_le(left, right) == np.array([False, True, True])).all()


def test_ge_by_lt() -> None:
    left, right = np.arange(3), 1
    assert (op.ge_by_lt(left, right) == np.array([False, True, True])).all()


def test_gt_by_ge() -> None:
    left, right = np.arange(3), 1
    assert (op.gt_by_ge(left, right) == np.array([False, False, True])).all()


def test_gt_by_le() -> None:
    left, right = np.arange(3), 1
    assert (op.gt_by_le(left, right) == np.array([False, False, True])).all()


def test_gt_by_lt() -> None:
    left, right = np.arange(3), 1
    assert (op.gt_by_lt(left, right) == np.array([False, False, True])).all()


def test_le_by_ge() -> None:
    left, right = np.arange(3), 1
    assert (op.le_by_ge(left, right) == np.array([True, True, False])).all()


def test_le_by_gt() -> None:
    left, right = np.arange(3), 1
    assert (op.le_by_gt(left, right) == np.array([True, True, False])).all()


def test_le_by_lt() -> None:
    left, right = np.arange(3), 1
    assert (op.le_by_lt(left, right) == np.array([True, True, False])).all()


def test_lt_by_ge() -> None:
    left, right = np.arange(3), 1
    assert (op.lt_by_ge(left, right) == np.array([True, False, False])).all()


def test_lt_by_gt() -> None:
    left, right = np.arange(3), 1
    assert (op.lt_by_gt(left, right) == np.array([True, False, False])).all()


def test_lt_by_le() -> None:
    left, right = np.arange(3), 1
    assert (op.lt_by_le(left, right) == np.array([True, False, False])).all()


def test_ne_by_eq() -> None:
    left, right = np.arange(3), 1
    assert (op.ne_by_eq(left, right) == np.array([True, False, True])).all()
