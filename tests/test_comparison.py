# standard library
from dataclasses import dataclass
from typing import Any


# dependencies
import numpy as np
from ndtools import TotalEquality, TotalOrdering


def test_TotalEquality_by_eq() -> None:
    class Even(TotalEquality):
        def __eq__(self, array: Any) -> Any:
            return array % 2 == 0

    left, right = np.arange(3), Even()
    assert all((left == right) == np.array([True, False, True]))
    assert all((right == left) == np.array([True, False, True]))
    assert all((left != right) == ~np.array([True, False, True]))
    assert all((right != left) == ~np.array([True, False, True]))


def test_TotalEquality_by_ne() -> None:
    class Even(TotalEquality):
        def __ne__(self, array: Any) -> Any:
            return array % 2 == 1

    left, right = np.arange(3), Even()
    assert all((left == right) == np.array([True, False, True]))
    assert all((right == left) == np.array([True, False, True]))
    assert all((left != right) == ~np.array([True, False, True]))
    assert all((right != left) == ~np.array([True, False, True]))


def test_TotalOrdering_by_ge() -> None:
    @dataclass
    class Range(TotalOrdering):
        lower: Any
        upper: Any

        def __eq__(self, array: Any) -> Any:
            return (array >= self.lower) & (array < self.upper)

        def __ge__(self, array: Any) -> Any:
            return array < self.upper

    left, right = np.arange(3), Range(1, 2)
    assert all((left == right) == np.array([False, True, False]))
    assert all((right == left) == np.array([False, True, False]))
    assert all((left != right) == np.array([True, False, True]))
    assert all((right != left) == np.array([True, False, True]))
    assert all((left >= right) == np.array([False, True, True]))
    assert all((right <= left) == np.array([False, True, True]))
    assert all((left > right) == np.array([False, False, True]))
    assert all((right < left) == np.array([False, False, True]))
    assert all((left <= right) == np.array([True, True, False]))
    assert all((right >= left) == np.array([True, True, False]))
    assert all((left < right) == np.array([True, False, False]))
    assert all((right > left) == np.array([True, False, False]))


def test_TotalOrdering_by_gt() -> None:
    @dataclass
    class Range(TotalOrdering):
        lower: Any
        upper: Any

        def __eq__(self, array: Any) -> Any:
            return (array >= self.lower) & (array < self.upper)

        def __gt__(self, array: Any) -> Any:
            return array < self.lower

    left, right = np.arange(3), Range(1, 2)
    assert all((left == right) == np.array([False, True, False]))
    assert all((right == left) == np.array([False, True, False]))
    assert all((left != right) == np.array([True, False, True]))
    assert all((right != left) == np.array([True, False, True]))
    assert all((left >= right) == np.array([False, True, True]))
    assert all((right <= left) == np.array([False, True, True]))
    assert all((left > right) == np.array([False, False, True]))
    assert all((right < left) == np.array([False, False, True]))
    assert all((left <= right) == np.array([True, True, False]))
    assert all((right >= left) == np.array([True, True, False]))
    assert all((left < right) == np.array([True, False, False]))
    assert all((right > left) == np.array([True, False, False]))


def test_TotalOrdering_by_le() -> None:
    @dataclass
    class Range(TotalOrdering):
        lower: Any
        upper: Any

        def __eq__(self, array: Any) -> Any:
            return (array >= self.lower) & (array < self.upper)

        def __le__(self, array: Any) -> Any:
            return array >= self.lower

    left, right = np.arange(3), Range(1, 2)
    assert all((left == right) == np.array([False, True, False]))
    assert all((right == left) == np.array([False, True, False]))
    assert all((left != right) == np.array([True, False, True]))
    assert all((right != left) == np.array([True, False, True]))
    assert all((left >= right) == np.array([False, True, True]))
    assert all((right <= left) == np.array([False, True, True]))
    assert all((left > right) == np.array([False, False, True]))
    assert all((right < left) == np.array([False, False, True]))
    assert all((left <= right) == np.array([True, True, False]))
    assert all((right >= left) == np.array([True, True, False]))
    assert all((left < right) == np.array([True, False, False]))
    assert all((right > left) == np.array([True, False, False]))


def test_TotalOrdering_by_lt() -> None:
    @dataclass
    class Range(TotalOrdering):
        lower: Any
        upper: Any

        def __eq__(self, array: Any) -> Any:
            return (array >= self.lower) & (array < self.upper)

        def __lt__(self, array: Any) -> Any:
            return array >= self.upper

    left, right = np.arange(3), Range(1, 2)
    assert all((left == right) == np.array([False, True, False]))
    assert all((right == left) == np.array([False, True, False]))
    assert all((left != right) == np.array([True, False, True]))
    assert all((right != left) == np.array([True, False, True]))
    assert all((left >= right) == np.array([False, True, True]))
    assert all((right <= left) == np.array([False, True, True]))
    assert all((left > right) == np.array([False, False, True]))
    assert all((right < left) == np.array([False, False, True]))
    assert all((left <= right) == np.array([True, True, False]))
    assert all((right >= left) == np.array([True, True, False]))
    assert all((left < right) == np.array([True, False, False]))
    assert all((right > left) == np.array([True, False, False]))
