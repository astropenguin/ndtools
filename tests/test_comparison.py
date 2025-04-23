# dependencies
import numpy as np
from ndtools import Apply, Match, Range
from numpy.char import isupper


def test_Apply() -> None:
    assert all((np.array(["A", "b"]) == Apply(isupper)) == np.array([True, False]))


def test_Match() -> None:
    assert all((np.array(["a", "aa"]) == Match("a+")) == np.array([True, True]))


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
