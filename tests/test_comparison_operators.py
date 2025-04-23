# standard library
from collections.abc import Callable
from typing import Any


# dependencies
import numpy as np
from ndtools.comparison.operators import eq, ge, gt, le, lt, ne


def test_operators_eqge() -> None:
    class Test:
        def __init__(self, value: Any) -> None:
            self.value = value

        def __eq__(self, array: Any) -> Any:
            return array == self.value

        def __ge__(self, array: Any) -> Any:
            return array <= self.value

    left, right = Test(1), np.arange(3)
    assert all(eq(left, right) == np.array([False, True, False]))
    assert all(ge(left, right) == np.array([True, True, False]))
    assert all(gt(left, right) == np.array([True, False, False]))
    assert all(le(left, right) == np.array([False, True, True]))
    assert all(lt(left, right) == np.array([False, False, True]))
    assert all(ne(left, right) == np.array([True, False, True]))


def test_operators_eqgt() -> None:
    class Test:
        def __init__(self, value: Any) -> None:
            self.value = value

        def __eq__(self, array: Any) -> Any:
            return array == self.value

        def __gt__(self, array: Any) -> Any:
            return array < self.value

    left, right = Test(1), np.arange(3)
    assert all(eq(left, right) == np.array([False, True, False]))
    assert all(ge(left, right) == np.array([True, True, False]))
    assert all(gt(left, right) == np.array([True, False, False]))
    assert all(le(left, right) == np.array([False, True, True]))
    assert all(lt(left, right) == np.array([False, False, True]))
    assert all(ne(left, right) == np.array([True, False, True]))


def test_operators_eqle() -> None:
    class Test:
        def __init__(self, value: Any) -> None:
            self.value = value

        def __eq__(self, array: Any) -> Any:
            return array == self.value

        def __le__(self, array: Any) -> Any:
            return array >= self.value

    left, right = Test(1), np.arange(3)
    assert all(eq(left, right) == np.array([False, True, False]))
    assert all(ge(left, right) == np.array([True, True, False]))
    assert all(gt(left, right) == np.array([True, False, False]))
    assert all(le(left, right) == np.array([False, True, True]))
    assert all(lt(left, right) == np.array([False, False, True]))
    assert all(ne(left, right) == np.array([True, False, True]))


def test_operators_eqlt() -> None:
    class Test:
        def __init__(self, value: Any) -> None:
            self.value = value

        def __eq__(self, array: Any) -> Any:
            return array == self.value

        def __lt__(self, array: Any) -> Any:
            return array > self.value

    left, right = Test(1), np.arange(3)
    assert all(eq(left, right) == np.array([False, True, False]))
    assert all(ge(left, right) == np.array([True, True, False]))
    assert all(gt(left, right) == np.array([True, False, False]))
    assert all(le(left, right) == np.array([False, True, True]))
    assert all(lt(left, right) == np.array([False, False, True]))
    assert all(ne(left, right) == np.array([True, False, True]))


def test_operators_gene() -> None:
    class Test:
        __eq__: Callable[..., Any]

        def __init__(self, value: Any) -> None:
            self.value = value

        def __ge__(self, array: Any) -> Any:
            return array <= self.value

        def __ne__(self, array: Any) -> Any:
            return array != self.value

    left, right = Test(1), np.arange(3)
    assert all(eq(left, right) == np.array([False, True, False]))
    assert all(ge(left, right) == np.array([True, True, False]))
    assert all(gt(left, right) == np.array([True, False, False]))
    assert all(le(left, right) == np.array([False, True, True]))
    assert all(lt(left, right) == np.array([False, False, True]))
    assert all(ne(left, right) == np.array([True, False, True]))


def test_operators_gtne() -> None:
    class Test:
        __eq__: Callable[..., Any]

        def __init__(self, value: Any) -> None:
            self.value = value

        def __gt__(self, array: Any) -> Any:
            return array < self.value

        def __ne__(self, array: Any) -> Any:
            return array != self.value

    left, right = Test(1), np.arange(3)
    assert all(eq(left, right) == np.array([False, True, False]))
    assert all(ge(left, right) == np.array([True, True, False]))
    assert all(gt(left, right) == np.array([True, False, False]))
    assert all(le(left, right) == np.array([False, True, True]))
    assert all(lt(left, right) == np.array([False, False, True]))
    assert all(ne(left, right) == np.array([True, False, True]))


def test_operators_lene() -> None:
    class Test:
        __eq__: Callable[..., Any]

        def __init__(self, value: Any) -> None:
            self.value = value

        def __le__(self, array: Any) -> Any:
            return array >= self.value

        def __ne__(self, array: Any) -> Any:
            return array != self.value

    left, right = Test(1), np.arange(3)
    assert all(eq(left, right) == np.array([False, True, False]))
    assert all(ge(left, right) == np.array([True, True, False]))
    assert all(gt(left, right) == np.array([True, False, False]))
    assert all(le(left, right) == np.array([False, True, True]))
    assert all(lt(left, right) == np.array([False, False, True]))
    assert all(ne(left, right) == np.array([True, False, True]))


def test_operators_ltne() -> None:
    class Test:
        __eq__: Callable[..., Any]

        def __init__(self, value: Any) -> None:
            self.value = value

        def __lt__(self, array: Any) -> Any:
            return array > self.value

        def __ne__(self, array: Any) -> Any:
            return array != self.value

    left, right = Test(1), np.arange(3)
    assert all(eq(left, right) == np.array([False, True, False]))
    assert all(ge(left, right) == np.array([True, True, False]))
    assert all(gt(left, right) == np.array([True, False, False]))
    assert all(le(left, right) == np.array([False, True, True]))
    assert all(lt(left, right) == np.array([False, False, True]))
    assert all(ne(left, right) == np.array([True, False, True]))
