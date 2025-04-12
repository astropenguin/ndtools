# standard library
from dataclasses import dataclass
from typing import Any


# dependencies
import numpy as np
from ndtools import total_equality, total_ordering


def test_total_equality_by_eq() -> None:
    @total_equality
    class Even:
        def __eq__(self, array: Any) -> Any:
            return ~((array % 2).astype(bool))

    left = np.arange(3)
    right: Any = Even()
    assert ((left == right) == np.array([True, False, True])).all()
    assert ((right == left) == np.array([True, False, True])).all()
    assert ((left != right) == ~np.array([True, False, True])).all()
    assert ((right != left) == ~np.array([True, False, True])).all()


def test_total_equality_by_ne() -> None:
    @total_equality
    class Even:
        def __ne__(self, array: Any) -> Any:
            return (array % 2).astype(bool)

    left = np.arange(3)
    right: Any = Even()
    assert ((left == right) == np.array([True, False, True])).all()
    assert ((right == left) == np.array([True, False, True])).all()
    assert ((left != right) == ~np.array([True, False, True])).all()
    assert ((right != left) == ~np.array([True, False, True])).all()


def test_total_ordering_by_ge() -> None:
    @dataclass
    @total_ordering
    class Interval:
        lower: Any
        upper: Any

        def __eq__(self, array: Any) -> Any:
            return (array >= self.lower) & (array < self.upper)

        def __ge__(self, array: Any) -> Any:
            return array < self.upper

    left = np.arange(3)
    right: Any = Interval(1, 2)
    assert ((left == right) == np.array([False, True, False])).all()
    assert ((right == left) == np.array([False, True, False])).all()
    assert ((left != right) == np.array([True, False, True])).all()
    assert ((right != left) == np.array([True, False, True])).all()
    assert ((left >= right) == np.array([False, True, True])).all()
    assert ((right <= left) == np.array([False, True, True])).all()
    assert ((left > right) == np.array([False, False, True])).all()
    assert ((right < left) == np.array([False, False, True])).all()
    assert ((left <= right) == np.array([True, True, False])).all()
    assert ((right >= left) == np.array([True, True, False])).all()
    assert ((left < right) == np.array([True, False, False])).all()
    assert ((right > left) == np.array([True, False, False])).all()


def test_total_ordering_by_gt() -> None:
    @dataclass
    @total_ordering
    class Interval:
        lower: Any
        upper: Any

        def __eq__(self, array: Any) -> Any:
            return (array >= self.lower) & (array < self.upper)

        def __gt__(self, array: Any) -> Any:
            return array < self.lower

    left = np.arange(3)
    right: Any = Interval(1, 2)
    assert ((left == right) == np.array([False, True, False])).all()
    assert ((right == left) == np.array([False, True, False])).all()
    assert ((left != right) == np.array([True, False, True])).all()
    assert ((right != left) == np.array([True, False, True])).all()
    assert ((left >= right) == np.array([False, True, True])).all()
    assert ((right <= left) == np.array([False, True, True])).all()
    assert ((left > right) == np.array([False, False, True])).all()
    assert ((right < left) == np.array([False, False, True])).all()
    assert ((left <= right) == np.array([True, True, False])).all()
    assert ((right >= left) == np.array([True, True, False])).all()
    assert ((left < right) == np.array([True, False, False])).all()
    assert ((right > left) == np.array([True, False, False])).all()


def test_total_ordering_by_le() -> None:
    @dataclass
    @total_ordering
    class Interval:
        lower: Any
        upper: Any

        def __eq__(self, array: Any) -> Any:
            return (array >= self.lower) & (array < self.upper)

        def __le__(self, array: Any) -> Any:
            return array >= self.lower

    left = np.arange(3)
    right: Any = Interval(1, 2)
    assert ((left == right) == np.array([False, True, False])).all()
    assert ((right == left) == np.array([False, True, False])).all()
    assert ((left != right) == np.array([True, False, True])).all()
    assert ((right != left) == np.array([True, False, True])).all()
    assert ((left >= right) == np.array([False, True, True])).all()
    assert ((right <= left) == np.array([False, True, True])).all()
    assert ((left > right) == np.array([False, False, True])).all()
    assert ((right < left) == np.array([False, False, True])).all()
    assert ((left <= right) == np.array([True, True, False])).all()
    assert ((right >= left) == np.array([True, True, False])).all()
    assert ((left < right) == np.array([True, False, False])).all()
    assert ((right > left) == np.array([True, False, False])).all()


def test_total_ordering_by_lt() -> None:
    @dataclass
    @total_ordering
    class Interval:
        lower: Any
        upper: Any

        def __eq__(self, array: Any) -> Any:
            return (array >= self.lower) & (array < self.upper)

        def __lt__(self, array: Any) -> Any:
            return array >= self.upper

    left = np.arange(3)
    right: Any = Interval(1, 2)
    assert ((left == right) == np.array([False, True, False])).all()
    assert ((right == left) == np.array([False, True, False])).all()
    assert ((left != right) == np.array([True, False, True])).all()
    assert ((right != left) == np.array([True, False, True])).all()
    assert ((left >= right) == np.array([False, True, True])).all()
    assert ((right <= left) == np.array([False, True, True])).all()
    assert ((left > right) == np.array([False, False, True])).all()
    assert ((right < left) == np.array([False, False, True])).all()
    assert ((left <= right) == np.array([True, True, False])).all()
    assert ((right >= left) == np.array([True, True, False])).all()
    assert ((left < right) == np.array([True, False, False])).all()
    assert ((right > left) == np.array([True, False, False])).all()
