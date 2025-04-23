__all__ = ["Apply", "Match", "Range"]


# standard libary
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal


# dependencies
import pandas as pd
from .comparable import Combinable, TotalEquality, TotalOrdering


@dataclass(frozen=True)
class Apply(Combinable, TotalEquality):
    """Equatable that applies a boolean function for multidimensional arrays.

    Args:
        func: Boolean function that takes ``func(array, *args, **kwargs)``.
        *args: Positional arguments to be passed to the function.
        **kwargs: Keyword arguments to be passed to the function.

    Examples:
        ::

            import numpy as np
            from ndtools import Apply
            from numpy.char import isupper

            np.array(["A", "b"]) == Apply(isupper)  # -> array([True, False])

    """

    func: Callable[..., Any]
    args: Any
    kwargs: Any

    def __init__(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__setattr__("func", func)
        super().__setattr__("args", args)
        super().__setattr__("kwargs", kwargs)

    def __eq__(self, array: Any) -> Any:
        return self.func(array, *self.args, **self.kwargs)

    def __repr__(self) -> str:
        return f"Apply({self.func}, *{self.args}, **{self.kwargs})"


@dataclass(frozen=True)
class Match(Combinable, TotalEquality):
    """Equatable that matches regular expression to each array element.

    It uses ``pandas.Series.str.fullmatch`` so the same options are available.

    Args:
        pat: Character sequence or regular expression.
        case: If True, case sensitive matching will be performed.
        flags: Regular expression flags, e.g. ``re.IGNORECASE``.
        na: Fill value for missing values.
            The default value depends on data type of the array.
            For object-dtype, ``numpy.nan`` will be used.
            For ``StringDtype``, ``pandas.NA`` will be used.

    Examples:
        ::

            import numpy as np
            from ndtools import Match

            np.array(["a", "aa"]) == Match("a+")  # -> array([True, True])

    """

    pat: str
    case: bool = True
    flags: int = 0
    na: Any = None

    def __eq__(self, array: Any) -> Any:
        return (
            pd.Series(array)  # type: ignore
            .str.fullmatch(self.pat, self.case, self.flags, self.na)
            .values
        )


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
