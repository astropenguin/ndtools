# standard library
from typing import Any as Any_


# dependencies
import numpy as np
from ndtools import (
    All,
    Any,
    Combinable,
    Equatable,
    Orderable,
    TotalEquality,
    TotalOrdering,
)


# helper functions
def eq(left: Any_, right: Any_, /) -> bool:
    return super(type(left), left).__eq__(right)


def Implemented(method: str) -> str:
    return f"{method} is implemented"


# test functions
def test_All() -> None:
    assert eq(All([0]) & 1, All([0, 1]))
    assert eq(All([0]) | 1, Any([All([0]), 1]))
    assert eq(All([0]) & All([1]), All([0, 1]))
    assert eq(All([0]) | All([1]), Any([All([0]), All([1])]))


def test_Any() -> None:
    assert eq(Any([0]) & 1, All([Any([0]), 1]))
    assert eq(Any([0]) | 1, Any([0, 1]))
    assert eq(Any([0]) & Any([1]), All([Any([0]), Any([1])]))
    assert eq(Any([0]) | Any([1]), Any([0, 1]))


def test_Combinable() -> None:
    class Test(Combinable):
        def __init__(self, value: Any_) -> None:
            self.value = value

        def __eq__(self, other: Any_) -> Any_:
            return self.value == other

    assert eq(Test(0) & 1, All([Test(0), 1]))
    assert eq(Test(0) | 1, Any([Test(0), 1]))


def test_Equatable() -> None:
    class Test(Equatable):
        def __eq__(self, other: Any_) -> Any_:
            return Implemented("__eq__")

        def __ne__(self, other: Any_) -> Any_:
            return Implemented("__ne__")

    assert (Test() == np.arange(3)) == Implemented("__eq__")
    assert (Test() != np.arange(3)) == Implemented("__ne__")

    assert (np.arange(3) == Test()) == Implemented("__eq__")
    assert (np.arange(3) != Test()) == Implemented("__ne__")


def test_Orderable() -> None:
    class Test(Orderable):
        def __eq__(self, other: Any_) -> Any_:
            return Implemented("__eq__")

        def __ge__(self, other: Any_) -> Any_:
            return Implemented("__ge__")

        def __gt__(self, other: Any_) -> Any_:
            return Implemented("__gt__")

        def __le__(self, other: Any_) -> Any_:
            return Implemented("__le__")

        def __lt__(self, other: Any_) -> Any_:
            return Implemented("__lt__")

        def __ne__(self, other: Any_) -> Any_:
            return Implemented("__ne__")

    assert (Test() == np.arange(3)) == Implemented("__eq__")
    assert (Test() >= np.arange(3)) == Implemented("__ge__")
    assert (Test() > np.arange(3)) == Implemented("__gt__")
    assert (Test() <= np.arange(3)) == Implemented("__le__")
    assert (Test() < np.arange(3)) == Implemented("__lt__")
    assert (Test() != np.arange(3)) == Implemented("__ne__")

    assert (np.arange(3) == Test()) == Implemented("__eq__")
    assert (np.arange(3) >= Test()) == Implemented("__le__")
    assert (np.arange(3) > Test()) == Implemented("__lt__")
    assert (np.arange(3) <= Test()) == Implemented("__ge__")
    assert (np.arange(3) < Test()) == Implemented("__gt__")
    assert (np.arange(3) != Test()) == Implemented("__ne__")


def test_TotalEquality_eq() -> None:
    class Test(TotalEquality):
        def __eq__(self, array: Any_) -> Any_:
            return array % 2 == 0

    left, right = np.arange(3), Test()
    assert all((left == right) == np.array([True, False, True]))
    assert all((right == left) == np.array([True, False, True]))
    assert all((left != right) == ~np.array([True, False, True]))
    assert all((right != left) == ~np.array([True, False, True]))


def test_TotalEquality_ne() -> None:
    class Test(TotalEquality):
        def __ne__(self, array: Any_) -> Any_:
            return array % 2 == 1

    left, right = np.arange(3), Test()
    assert all((left == right) == np.array([True, False, True]))
    assert all((right == left) == np.array([True, False, True]))
    assert all((left != right) == ~np.array([True, False, True]))
    assert all((right != left) == ~np.array([True, False, True]))


def test_TotalOrdering_eqge() -> None:
    class Test(TotalOrdering):
        def __init__(self, value: Any_) -> None:
            self.value = value

        def __eq__(self, array: Any_) -> Any_:
            return array == self.value

        def __ge__(self, array: Any_) -> Any_:
            return array <= self.value

    left, right = Test(1), np.arange(3)
    assert all((left == right) == np.array([False, True, False]))
    assert all((right == left) == np.array([False, True, False]))
    assert all((left >= right) == np.array([True, True, False]))
    assert all((right <= left) == np.array([True, True, False]))
    assert all((left > right) == np.array([True, False, False]))
    assert all((right < left) == np.array([True, False, False]))
    assert all((left <= right) == np.array([False, True, True]))
    assert all((right >= left) == np.array([False, True, True]))
    assert all((left < right) == np.array([False, False, True]))
    assert all((right > left) == np.array([False, False, True]))
    assert all((left != right) == np.array([True, False, True]))
    assert all((right != left) == np.array([True, False, True]))


def test_TotalOrdering_eqgt() -> None:
    class Test(TotalOrdering):
        def __init__(self, value: Any_) -> None:
            self.value = value

        def __eq__(self, array: Any_) -> Any_:
            return array == self.value

        def __gt__(self, array: Any_) -> Any_:
            return array < self.value

    left, right = Test(1), np.arange(3)
    assert all((left == right) == np.array([False, True, False]))
    assert all((right == left) == np.array([False, True, False]))
    assert all((left >= right) == np.array([True, True, False]))
    assert all((right <= left) == np.array([True, True, False]))
    assert all((left > right) == np.array([True, False, False]))
    assert all((right < left) == np.array([True, False, False]))
    assert all((left <= right) == np.array([False, True, True]))
    assert all((right >= left) == np.array([False, True, True]))
    assert all((left < right) == np.array([False, False, True]))
    assert all((right > left) == np.array([False, False, True]))
    assert all((left != right) == np.array([True, False, True]))
    assert all((right != left) == np.array([True, False, True]))


def test_TotalOrdering_eqle() -> None:
    class Test(TotalOrdering):
        def __init__(self, value: Any_) -> None:
            self.value = value

        def __eq__(self, array: Any_) -> Any_:
            return array == self.value

        def __le__(self, array: Any_) -> Any_:
            return array >= self.value

    left, right = Test(1), np.arange(3)
    assert all((left == right) == np.array([False, True, False]))
    assert all((right == left) == np.array([False, True, False]))
    assert all((left >= right) == np.array([True, True, False]))
    assert all((right <= left) == np.array([True, True, False]))
    assert all((left > right) == np.array([True, False, False]))
    assert all((right < left) == np.array([True, False, False]))
    assert all((left <= right) == np.array([False, True, True]))
    assert all((right >= left) == np.array([False, True, True]))
    assert all((left < right) == np.array([False, False, True]))
    assert all((right > left) == np.array([False, False, True]))
    assert all((left != right) == np.array([True, False, True]))
    assert all((right != left) == np.array([True, False, True]))


def test_TotalOrdering_eqlt() -> None:
    class Test(TotalOrdering):
        def __init__(self, value: Any_) -> None:
            self.value = value

        def __eq__(self, array: Any_) -> Any_:
            return array == self.value

        def __lt__(self, array: Any_) -> Any_:
            return array > self.value

    left, right = Test(1), np.arange(3)
    assert all((left == right) == np.array([False, True, False]))
    assert all((right == left) == np.array([False, True, False]))
    assert all((left >= right) == np.array([True, True, False]))
    assert all((right <= left) == np.array([True, True, False]))
    assert all((left > right) == np.array([True, False, False]))
    assert all((right < left) == np.array([True, False, False]))
    assert all((left <= right) == np.array([False, True, True]))
    assert all((right >= left) == np.array([False, True, True]))
    assert all((left < right) == np.array([False, False, True]))
    assert all((right > left) == np.array([False, False, True]))
    assert all((left != right) == np.array([True, False, True]))
    assert all((right != left) == np.array([True, False, True]))


def test_TotalOrdering_gene() -> None:
    class Test(TotalOrdering):
        def __init__(self, value: Any_) -> None:
            self.value = value

        def __ge__(self, array: Any_) -> Any_:
            return array <= self.value

        def __ne__(self, array: Any_) -> Any_:
            return array != self.value

    left, right = Test(1), np.arange(3)
    assert all((left == right) == np.array([False, True, False]))
    assert all((right == left) == np.array([False, True, False]))
    assert all((left >= right) == np.array([True, True, False]))
    assert all((right <= left) == np.array([True, True, False]))
    assert all((left > right) == np.array([True, False, False]))
    assert all((right < left) == np.array([True, False, False]))
    assert all((left <= right) == np.array([False, True, True]))
    assert all((right >= left) == np.array([False, True, True]))
    assert all((left < right) == np.array([False, False, True]))
    assert all((right > left) == np.array([False, False, True]))
    assert all((left != right) == np.array([True, False, True]))
    assert all((right != left) == np.array([True, False, True]))


def test_TotalOrdering_gtne() -> None:
    class Test(TotalOrdering):
        def __init__(self, value: Any_) -> None:
            self.value = value

        def __gt__(self, array: Any_) -> Any_:
            return array < self.value

        def __ne__(self, array: Any_) -> Any_:
            return array != self.value

    left, right = Test(1), np.arange(3)
    assert all((left == right) == np.array([False, True, False]))
    assert all((right == left) == np.array([False, True, False]))
    assert all((left >= right) == np.array([True, True, False]))
    assert all((right <= left) == np.array([True, True, False]))
    assert all((left > right) == np.array([True, False, False]))
    assert all((right < left) == np.array([True, False, False]))
    assert all((left <= right) == np.array([False, True, True]))
    assert all((right >= left) == np.array([False, True, True]))
    assert all((left < right) == np.array([False, False, True]))
    assert all((right > left) == np.array([False, False, True]))
    assert all((left != right) == np.array([True, False, True]))
    assert all((right != left) == np.array([True, False, True]))


def test_TotalOrdering_lene() -> None:
    class Test(TotalOrdering):
        def __init__(self, value: Any_) -> None:
            self.value = value

        def __le__(self, array: Any_) -> Any_:
            return array >= self.value

        def __ne__(self, array: Any_) -> Any_:
            return array != self.value

    left, right = Test(1), np.arange(3)
    assert all((left == right) == np.array([False, True, False]))
    assert all((right == left) == np.array([False, True, False]))
    assert all((left >= right) == np.array([True, True, False]))
    assert all((right <= left) == np.array([True, True, False]))
    assert all((left > right) == np.array([True, False, False]))
    assert all((right < left) == np.array([True, False, False]))
    assert all((left <= right) == np.array([False, True, True]))
    assert all((right >= left) == np.array([False, True, True]))
    assert all((left < right) == np.array([False, False, True]))
    assert all((right > left) == np.array([False, False, True]))
    assert all((left != right) == np.array([True, False, True]))
    assert all((right != left) == np.array([True, False, True]))


def test_TotalOrdering_ltne() -> None:
    class Test(TotalOrdering):
        def __init__(self, value: Any_) -> None:
            self.value = value

        def __lt__(self, array: Any_) -> Any_:
            return array > self.value

        def __ne__(self, array: Any_) -> Any_:
            return array != self.value

    left, right = Test(1), np.arange(3)
    assert all((left == right) == np.array([False, True, False]))
    assert all((right == left) == np.array([False, True, False]))
    assert all((left >= right) == np.array([True, True, False]))
    assert all((right <= left) == np.array([True, True, False]))
    assert all((left > right) == np.array([True, False, False]))
    assert all((right < left) == np.array([True, False, False]))
    assert all((left <= right) == np.array([False, True, True]))
    assert all((right >= left) == np.array([False, True, True]))
    assert all((left < right) == np.array([False, False, True]))
    assert all((right > left) == np.array([False, False, True]))
    assert all((left != right) == np.array([True, False, True]))
    assert all((right != left) == np.array([True, False, True]))
