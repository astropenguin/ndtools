# dependencies
import numpy as np
from ndtools import ANY, NEVER
from ndtools.comparison.builtins import AnyType, NeverType


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
