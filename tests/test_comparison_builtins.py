# dependencies
import numpy as np
from ndtools import ANY, NEVER, Match, Range, Where
from ndtools.comparison.builtins import AnyType, NeverType
from numpy.char import isupper


def test_ANY() -> None:
    assert all((np.arange(3) == ANY) == np.array([True, True, True]))


def test_NEVER() -> None:
    assert all((np.arange(3) == NEVER) == np.array([False, False, False]))


def test_AnyType() -> None:
    assert AnyType() is AnyType()
    assert all((np.arange(3) == AnyType()) == np.array([True, True, True]))


def test_NeverType() -> None:
    assert NeverType() is NeverType()
    assert all((np.arange(3) == NeverType()) == np.array([False, False, False]))


def test_Match() -> None:
    assert all((np.array(["a", "aa"]) == Match("a+")) == np.array([True, True]))


def test_Range_eq() -> None:
    data = np.arange(3)
    assert all((data == Range(1, 2, "[]")) == np.array([False, True, True]))
    assert all((data == Range(1, 2, "[)")) == np.array([False, True, False]))
    assert all((data == Range(1, 2, "(]")) == np.array([False, False, True]))
    assert all((data == Range(1, 2, "()")) == np.array([False, False, False]))
    assert all((data == Range(None, 2, "[]")) == np.array([True, True, True]))
    assert all((data == Range(None, 2, "[)")) == np.array([True, True, False]))
    assert all((data == Range(None, 2, "(]")) == np.array([True, True, True]))
    assert all((data == Range(None, 2, "()")) == np.array([True, True, False]))
    assert all((data == Range(1, None, "[]")) == np.array([False, True, True]))
    assert all((data == Range(1, None, "[)")) == np.array([False, True, True]))
    assert all((data == Range(1, None, "(]")) == np.array([False, False, True]))
    assert all((data == Range(1, None, "()")) == np.array([False, False, True]))
    assert all((data == Range(None, None, "[]")) == np.array([True, True, True]))
    assert all((data == Range(None, None, "[)")) == np.array([True, True, True]))
    assert all((data == Range(None, None, "(]")) == np.array([True, True, True]))
    assert all((data == Range(None, None, "()")) == np.array([True, True, True]))


def test_Range_ge() -> None:
    data = np.arange(3)
    assert all((data >= Range(1, 2, "[]")) == np.array([False, True, True]))
    assert all((data >= Range(1, 2, "[)")) == np.array([False, True, True]))
    assert all((data >= Range(1, 2, "(]")) == np.array([False, False, True]))
    assert all((data >= Range(1, 2, "()")) == np.array([False, False, True]))
    assert all((data >= Range(None, 2, "[]")) == np.array([True, True, True]))
    assert all((data >= Range(None, 2, "[)")) == np.array([True, True, True]))
    assert all((data >= Range(None, 2, "(]")) == np.array([True, True, True]))
    assert all((data >= Range(None, 2, "()")) == np.array([True, True, True]))
    assert all((data >= Range(1, None, "[]")) == np.array([False, True, True]))
    assert all((data >= Range(1, None, "[)")) == np.array([False, True, True]))
    assert all((data >= Range(1, None, "(]")) == np.array([False, False, True]))
    assert all((data >= Range(1, None, "()")) == np.array([False, False, True]))
    assert all((data >= Range(None, None, "[]")) == np.array([True, True, True]))
    assert all((data >= Range(None, None, "[)")) == np.array([True, True, True]))
    assert all((data >= Range(None, None, "(]")) == np.array([True, True, True]))
    assert all((data >= Range(None, None, "()")) == np.array([True, True, True]))


def test_Range_gt() -> None:
    data = np.arange(3)
    assert all((data > Range(1, 2, "[]")) == np.array([False, False, False]))
    assert all((data > Range(1, 2, "[)")) == np.array([False, False, True]))
    assert all((data > Range(1, 2, "(]")) == np.array([False, False, False]))
    assert all((data > Range(1, 2, "()")) == np.array([False, False, True]))
    assert all((data > Range(None, 2, "[]")) == np.array([False, False, False]))
    assert all((data > Range(None, 2, "[)")) == np.array([False, False, True]))
    assert all((data > Range(None, 2, "(]")) == np.array([False, False, False]))
    assert all((data > Range(None, 2, "()")) == np.array([False, False, True]))
    assert all((data > Range(1, None, "[]")) == np.array([False, False, False]))
    assert all((data > Range(1, None, "[)")) == np.array([False, False, False]))
    assert all((data > Range(1, None, "(]")) == np.array([False, False, False]))
    assert all((data > Range(1, None, "()")) == np.array([False, False, False]))
    assert all((data > Range(None, None, "[]")) == np.array([False, False, False]))
    assert all((data > Range(None, None, "[)")) == np.array([False, False, False]))
    assert all((data > Range(None, None, "(]")) == np.array([False, False, False]))
    assert all((data > Range(None, None, "()")) == np.array([False, False, False]))


def test_Range_le() -> None:
    data = np.arange(3)
    assert all((data <= Range(1, 2, "[]")) == np.array([True, True, True]))
    assert all((data <= Range(1, 2, "[)")) == np.array([True, True, False]))
    assert all((data <= Range(1, 2, "(]")) == np.array([True, True, True]))
    assert all((data <= Range(1, 2, "()")) == np.array([True, True, False]))
    assert all((data <= Range(None, 2, "[]")) == np.array([True, True, True]))
    assert all((data <= Range(None, 2, "[)")) == np.array([True, True, False]))
    assert all((data <= Range(None, 2, "(]")) == np.array([True, True, True]))
    assert all((data <= Range(None, 2, "()")) == np.array([True, True, False]))
    assert all((data <= Range(1, None, "[]")) == np.array([True, True, True]))
    assert all((data <= Range(1, None, "[)")) == np.array([True, True, True]))
    assert all((data <= Range(1, None, "(]")) == np.array([True, True, True]))
    assert all((data <= Range(1, None, "()")) == np.array([True, True, True]))
    assert all((data <= Range(None, None, "[]")) == np.array([True, True, True]))
    assert all((data <= Range(None, None, "[)")) == np.array([True, True, True]))
    assert all((data <= Range(None, None, "(]")) == np.array([True, True, True]))
    assert all((data <= Range(None, None, "()")) == np.array([True, True, True]))


def test_Range_lt() -> None:
    data = np.arange(3)
    assert all((data < Range(1, 2, "[]")) == np.array([True, False, False]))
    assert all((data < Range(1, 2, "[)")) == np.array([True, False, False]))
    assert all((data < Range(1, 2, "(]")) == np.array([True, True, False]))
    assert all((data < Range(1, 2, "()")) == np.array([True, True, False]))
    assert all((data < Range(None, 2, "[]")) == np.array([False, False, False]))
    assert all((data < Range(None, 2, "[)")) == np.array([False, False, False]))
    assert all((data < Range(None, 2, "(]")) == np.array([False, False, False]))
    assert all((data < Range(None, 2, "()")) == np.array([False, False, False]))
    assert all((data < Range(1, None, "[]")) == np.array([True, False, False]))
    assert all((data < Range(1, None, "[)")) == np.array([True, False, False]))
    assert all((data < Range(1, None, "(]")) == np.array([True, True, False]))
    assert all((data < Range(1, None, "()")) == np.array([True, True, False]))
    assert all((data < Range(None, None, "[]")) == np.array([False, False, False]))
    assert all((data < Range(None, None, "[)")) == np.array([False, False, False]))
    assert all((data < Range(None, None, "(]")) == np.array([False, False, False]))
    assert all((data < Range(None, None, "()")) == np.array([False, False, False]))


def test_Range_ne() -> None:
    data = np.arange(3)
    assert all((data != Range(1, 2, "[]")) == np.array([True, False, False]))
    assert all((data != Range(1, 2, "[)")) == np.array([True, False, True]))
    assert all((data != Range(1, 2, "(]")) == np.array([True, True, False]))
    assert all((data != Range(1, 2, "()")) == np.array([True, True, True]))
    assert all((data != Range(None, 2, "[]")) == np.array([False, False, False]))
    assert all((data != Range(None, 2, "[)")) == np.array([False, False, True]))
    assert all((data != Range(None, 2, "(]")) == np.array([False, False, False]))
    assert all((data != Range(None, 2, "()")) == np.array([False, False, True]))
    assert all((data != Range(1, None, "[]")) == np.array([True, False, False]))
    assert all((data != Range(1, None, "[)")) == np.array([True, False, False]))
    assert all((data != Range(1, None, "(]")) == np.array([True, True, False]))
    assert all((data != Range(1, None, "()")) == np.array([True, True, False]))
    assert all((data != Range(None, None, "[]")) == np.array([False, False, False]))
    assert all((data != Range(None, None, "[)")) == np.array([False, False, False]))
    assert all((data != Range(None, None, "(]")) == np.array([False, False, False]))
    assert all((data != Range(None, None, "()")) == np.array([False, False, False]))


def test_Where() -> None:
    assert all((np.array(["A", "b"]) == Where(isupper)) == np.array([True, False]))
