# standard library
from typing import Any

# dependencies
import numpy as np
from ndtools import Equatable, Orderable


def Implemented(method: str) -> str:
    return f"{method} is implemented"


def test_Equatable() -> None:
    class Test(Equatable):
        def __eq__(self, other: Any) -> Any:
            return Implemented("__eq__")

        def __ne__(self, other: Any) -> Any:
            return Implemented("__ne__")

    assert (Test() == np.arange(3)) == Implemented("__eq__")
    assert (Test() != np.arange(3)) == Implemented("__ne__")

    assert (np.arange(3) == Test()) == Implemented("__eq__")
    assert (np.arange(3) != Test()) == Implemented("__ne__")


def test_Orderable() -> None:
    class Test(Orderable):
        def __eq__(self, other: Any) -> Any:
            return Implemented("__eq__")

        def __ge__(self, other: Any) -> Any:
            return Implemented("__ge__")

        def __gt__(self, other: Any) -> Any:
            return Implemented("__gt__")

        def __le__(self, other: Any) -> Any:
            return Implemented("__le__")

        def __lt__(self, other: Any) -> Any:
            return Implemented("__lt__")

        def __ne__(self, other: Any) -> Any:
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
