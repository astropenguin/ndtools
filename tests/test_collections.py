# dependencies
import numpy as np
from ndtools import Range


def test_Range_eq() -> None:
    data = np.arange(3)
    assert all((data == Range(1, 2, "[]")) == np.array([False, True, True]))
    assert all((data == Range(1, 2, "[)")) == np.array([False, True, False]))
    assert all((data == Range(1, 2, "(]")) == np.array([False, False, True]))
    assert all((data == Range(1, 2, "()")) == np.array([False, False, False]))


def test_Range_ge() -> None:
    data = np.arange(3)
    assert all((data >= Range(1, 2, "[]")) == np.array([False, True, True]))
    assert all((data >= Range(1, 2, "[)")) == np.array([False, True, True]))
    assert all((data >= Range(1, 2, "(]")) == np.array([False, False, True]))
    assert all((data >= Range(1, 2, "()")) == np.array([False, False, True]))


def test_Range_gt() -> None:
    data = np.arange(3)
    assert all((data > Range(1, 2, "[]")) == np.array([False, False, False]))
    assert all((data > Range(1, 2, "[)")) == np.array([False, False, True]))
    assert all((data > Range(1, 2, "(]")) == np.array([False, False, False]))
    assert all((data > Range(1, 2, "()")) == np.array([False, False, True]))


def test_Range_le() -> None:
    data = np.arange(3)
    assert all((data <= Range(1, 2, "[]")) == np.array([True, True, True]))
    assert all((data <= Range(1, 2, "[)")) == np.array([True, True, False]))
    assert all((data <= Range(1, 2, "(]")) == np.array([True, True, True]))
    assert all((data <= Range(1, 2, "()")) == np.array([True, True, False]))


def test_Range_lt() -> None:
    data = np.arange(3)
    assert all((data < Range(1, 2, "[]")) == np.array([True, False, False]))
    assert all((data < Range(1, 2, "[)")) == np.array([True, False, False]))
    assert all((data < Range(1, 2, "(]")) == np.array([True, True, False]))
    assert all((data < Range(1, 2, "()")) == np.array([True, True, False]))


def test_Range_ne() -> None:
    data = np.arange(3)
    assert all((data != Range(1, 2, "[]")) == np.array([True, False, False]))
    assert all((data != Range(1, 2, "[)")) == np.array([True, False, True]))
    assert all((data != Range(1, 2, "(]")) == np.array([True, True, False]))
    assert all((data != Range(1, 2, "()")) == np.array([True, True, True]))
