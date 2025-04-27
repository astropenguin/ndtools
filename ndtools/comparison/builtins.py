__all__ = ["ANY", "NEVER", "AnyType", "NeverType", "Match", "Range", "Where"]


# standard library
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any as Any_, Literal


# dependencies
import pandas as pd
from typing_extensions import Self
from .comparables import Combinable, Equatable, Orderable


class AnyType(Combinable, Equatable):
    """Comparable that is always evaluated as True.

    It is singleton and all instances created by ``AnyType()``
    are thus identical. ndtools provides it as ``ndtools.ANY``.

    Examples:
        ::

            import numpy as np
            from ndtools import ANY

            np.arange(3) == ANY  # -> array([True, True, True])

    """

    _ANY: Self

    def __new__(cls) -> Self:
        if not hasattr(cls, "_ANY"):
            cls._ANY = super().__new__(cls)

        return cls._ANY

    def __eq__(self, other: Any_) -> Any_:
        return (other == other) | True

    def __repr__(self) -> str:
        return "ANY"


class NeverType(Combinable, Equatable):
    """Comparable that is always evaluated as False.

    It is singleton and all instances created by ``NeverType()``
    are thus identical. ndtools provides it as ``ndtools.NEVER``.

    Examples:
        ::

            import numpy as np
            from ndtools import NEVER

            np.arange(3) == NEVER  # -> array([False, False, False])

    """

    _NEVER: Self

    def __new__(cls) -> Self:
        if not hasattr(cls, "_NEVER"):
            cls._NEVER = super().__new__(cls)

        return cls._NEVER

    def __eq__(self, other: Any_) -> Any_:
        return (other != other) & False

    def __repr__(self) -> str:
        return "NEVER"


ANY = AnyType()
"""Comparable that is always evaluated as True.

Examples:
    ::

        import numpy as np
        from ndtools import ANY

        np.arange(3) == ANY  # -> array([True, True, True])

"""


NEVER = NeverType()
"""Comparable that is always evaluated as False.

Examples:
    ::

        import numpy as np
        from ndtools import NEVER

        np.arange(3) == NEVER  # -> array([False, False, False])

"""


@dataclass(frozen=True)
class Match(Combinable, Equatable):
    """Comparable that matches regular expression to each array element.

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
    """Character sequence or regular expression."""

    case: bool = True
    """If True, case sensitive matching will be performed."""

    flags: int = 0
    """Regular expression flags, e.g. ``re.IGNORECASE``."""

    na: Any_ = None
    """Fill value for missing values."""

    def __eq__(self, other: Any_) -> Any_:
        return (
            pd.Series(other)  # type: ignore
            .str.fullmatch(self.pat, self.case, self.flags, self.na)
            .values
        )


@dataclass(frozen=True)
class Range(Combinable, Orderable):
    """Comparable that implements equivalence with a certain range.

    Args:
        lower: Lower value of the range. If ``None`` is specified,
            then the lower value comparison will be skipped.
        lower: Upper value of the range. If ``None`` is specified,
            then the upper value comparison will be skipped.
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

            np.arange(3) == Range(None, 2)  # -> array([True, True, False])
            np.arange(3) == Range(1, None)  # -> array([False, True, True])
            np.arange(3) == Range(None, None)  # -> array([True, True, True])

    """

    lower: Any_
    """Lower value of the range."""

    upper: Any_
    """Upper value of the range."""

    bounds: Literal["[]", "[)", "(]", "()"] = "[)"
    """Type of bounds of the range."""

    @property
    def is_lower_open(self) -> bool:
        """Check if the lower bound is open."""
        return self.bounds[0] == "("

    @property
    def is_lower_closed(self) -> bool:
        """Check if the lower bound is closed."""
        return self.bounds[0] == "["

    @property
    def is_upper_open(self) -> bool:
        """Check if the upper bound is open."""
        return self.bounds[1] == ")"

    @property
    def is_upper_closed(self) -> bool:
        """Check if the upper bound is closed."""
        return self.bounds[1] == "]"

    def __eq__(self, other: Any_) -> Any_:
        if self.lower is None and self.upper is None:
            return other == ANY

        if self.lower is None and self.upper is not None and self.is_upper_closed:
            return other <= self.upper

        if self.lower is None and self.upper is not None and self.is_upper_open:
            return other < self.upper

        if self.lower is not None and self.upper is None and self.is_lower_closed:
            return other >= self.lower

        if self.lower is not None and self.upper is None and self.is_lower_open:
            return other > self.lower

        if self.is_lower_closed and self.is_upper_closed:
            return (other >= self.lower) & (other <= self.upper)

        if self.is_lower_closed and self.is_upper_open:
            return (other >= self.lower) & (other < self.upper)

        if self.is_lower_open and self.is_upper_closed:
            return (other > self.lower) & (other <= self.upper)

        if self.is_lower_open and self.is_upper_open:
            return (other > self.lower) & (other < self.upper)

        raise ValueError("Bounds must be either [], [), (], or ().")

    def __gt__(self, other: Any_) -> Any_:
        if self.lower is None:
            return other == NEVER

        if self.is_lower_closed:
            return other < self.lower

        if self.is_lower_open:
            return other <= self.lower

        raise ValueError("Bounds must be either [], [), (], or ().")

    def __lt__(self, other: Any_) -> Any_:
        if self.upper is None:
            return other == NEVER

        if self.is_upper_closed:
            return other > self.upper

        if self.is_upper_open:
            return other >= self.upper

        raise ValueError("Bounds must be either [], [), (], or ().")

    def __repr__(self) -> str:
        return f"{self.bounds[0]}{self.lower}, {self.upper}{self.bounds[1]}"


@dataclass(frozen=True)
class Where(Combinable, Equatable):
    """Comparable that applies a boolean function for multidimensional arrays.

    Args:
        func: Boolean function that takes ``func(array, *args, **kwargs)``.
        *args: Positional arguments to be passed to the function.
        **kwargs: Keyword arguments to be passed to the function.

    Examples:
        ::

            import numpy as np
            from ndtools import Where
            from numpy.char import isupper

            np.array(["A", "b"]) == Where(isupper)  # -> array([True, False])

    """

    func: Callable[..., Any_]
    """Boolean function that takes ``func(array, *args, **kwargs)``."""

    args: Any_
    """Positional arguments to be passed to the function."""

    kwargs: Any_
    """Keyword arguments to be passed to the function."""

    def __init__(self, func: Callable[..., Any_], *args: Any_, **kwargs: Any_) -> None:
        super().__setattr__("func", func)
        super().__setattr__("args", args)
        super().__setattr__("kwargs", kwargs)

    def __eq__(self, other: Any_) -> Any_:
        return self.func(other, *self.args, **self.kwargs)

    def __repr__(self) -> str:
        return f"Apply({self.func}, *{self.args}, **{self.kwargs})"
