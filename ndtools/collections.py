__all__ = ["Range"]


# standard libary
from dataclasses import dataclass
from typing import Any, Literal


# dependencies
from .combination import Combinable
from .comparison import TotalOrdering


@dataclass(frozen=True)
class Range(Combinable, TotalOrdering):
    """Equitable that implements equivalence with a certain range.

    Args:
        lower: Lower value of the range.
        upper: Upper value of the range.
        bounds: Type of bounds of the range.
            ``[]``: Lower-closed and upper-closed.
            ``[)``: Lower-closed and upper-open (default).
            ``(]``: Lower-open and upper-closed.
            ``()``: Lower-open and upper-open.

    Examples:
        ::

            import numpy as np
            from ndtools import Range

            np.arange(3) == Range(1, 2) # -> array([False, True, False])
            np.arange(3) < Range(1, 2)  # -> array([True, False, False])
            np.arange(3) > Range(1, 2)  # -> array([False, False, True])

    """

    lower: Any
    upper: Any
    bounds: Literal["[]", "[)", "(]", "()"] = "[)"

    def __eq__(self, array: Any) -> Any:
        if self.bounds == "[]":
            return (array >= self.lower) & (array <= self.upper)

        if self.bounds == "[)":
            return (array >= self.lower) & (array < self.upper)

        if self.bounds == "(]":
            return (array > self.lower) & (array <= self.upper)

        if self.bounds == "()":
            return (array > self.lower) & (array < self.upper)

        raise ValueError("Bounds must be either [], [), (], or [].")

    def __ge__(self, array: Any) -> Any:
        if self.bounds == "[)" or self.bounds == "()":
            return array < self.upper

        if self.bounds == "[]" or self.bounds == "(]":
            return array <= self.upper

        raise ValueError("Bounds must be either [], [), (], or [].")

    def __gt__(self, array: Any) -> Any:
        if self.bounds == "[]" or self.bounds == "[)":
            return array < self.lower

        if self.bounds == "(]" or self.bounds == "()":
            return array <= self.lower

        raise ValueError("Bounds must be either [], [), (], or [].")

    def __le__(self, array: Any) -> Any:
        return ~(self > array)

    def __lt__(self, array: Any) -> Any:
        return ~(self >= array)

    def __ne__(self, array: Any) -> Any:
        return ~(self == array)

    def __repr__(self) -> str:
        return f"{self.bounds[0]}{self.lower}, {self.upper}{self.bounds[1]}"
