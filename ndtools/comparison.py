__all__ = ["total_equality", "total_ordering"]


# standard library
from typing import Any, TypeVar


# dependencies
import numpy as np
from . import operators as op


# type hints
T = TypeVar("T")


# constants
MISSINGS_EQUALITY = {
    "__eq__": {
        "__ne__": op.ne_by_eq,
    },
    "__ne__": {
        "__eq__": op.eq_by_ne,
    },
}
MISSINGS_ORDERING = {
    "__ge__": {
        "__gt__": op.gt_by_ge,
        "__le__": op.le_by_ge,
        "__lt__": op.lt_by_ge,
    },
    "__gt__": {
        "__ge__": op.ge_by_gt,
        "__le__": op.le_by_gt,
        "__lt__": op.lt_by_gt,
    },
    "__le__": {
        "__gt__": op.gt_by_le,
        "__ge__": op.ge_by_le,
        "__lt__": op.lt_by_le,
    },
    "__lt__": {
        "__gt__": op.gt_by_lt,
        "__ge__": op.ge_by_lt,
        "__le__": op.le_by_lt,
    },
}


def has_userattr(obj: Any, name: str, /) -> bool:
    """Check if an object has a used-defined attribute with given name."""
    return getattr(obj, name, None) is not getattr(object, name, None)


def total_equality(cls: type[T], /) -> type[T]:
    """Class decorator that fills in missing multidimensional equality methods.

    Args:
        cls: Class to be decorated.

    Returns:
        The same class with missing multidimensional equality methods.

    Examples:
        ::

            import numpy as np
            from ndtools import total_equality


            @total_equality
            class Even:
                def __eq__(self, array):
                    return ~((array % 2).astype(bool))


            result = (np.arange(3) == Even())
            expected = np.array([True, False, True])
            assert (result == expected).all()

    """
    defined = [name for name in MISSINGS_EQUALITY if has_userattr(cls, name)]

    if not defined:
        raise ValueError("Define at least one equality operator (==, !=).")

    for name, operator in MISSINGS_EQUALITY[defined[0]].items():
        if not has_userattr(cls, name):
            setattr(cls, name, operator)

    def __array_ufunc__(
        self: Any,
        ufunc: np.ufunc,
        method: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
        if ufunc is np.equal:
            return self == inputs[0]
        elif ufunc is np.not_equal:
            return self != inputs[0]
        else:
            return NotImplemented

    setattr(cls, "__array_ufunc__", __array_ufunc__)
    return cls


def total_ordering(cls: type[T], /) -> type[T]:
    """Class decorator that fills in missing multidimensional ordering methods.

    Args:
        cls: Class to be decorated.

    Returns:
        The same class with missing multidimensional ordering methods.

    Examples:
        ::

            import numpy as np
            from dataclasses import dataclass
            from ndtools import total_ordering


            @dataclass
            @total_ordering
            class Interval:
                lower: float
                upper: float

                def __eq__(self, array):
                    return (array >= self.lower) & (array < self.upper)

                def __ge__(self, array):
                    return array < self.upper


            result = (np.arange(3) == Interval(1, 2))
            expected = np.array([False, True, False])
            assert (result == expected).all()

            result = (np.arange(3) < Interval(1, 2))
            expected = np.array([True, False, False])
            assert (result == expected).all()

    """
    cls = total_equality(cls)
    defined = [name for name in MISSINGS_ORDERING if has_userattr(cls, name)]

    if not defined:
        raise ValueError("Define at least one ordering operator (>=, >, <=, <).")

    for name, operator in MISSINGS_ORDERING[defined[0]].items():
        if not has_userattr(cls, name):
            setattr(cls, name, operator)

    def __array_ufunc__(
        self: Any,
        ufunc: np.ufunc,
        method: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
        if ufunc is np.equal:
            return self == inputs[0]
        elif ufunc is np.greater:
            return self < inputs[0]
        elif ufunc is np.greater_equal:
            return self <= inputs[0]
        elif ufunc is np.less:
            return self > inputs[0]
        elif ufunc is np.less_equal:
            return self >= inputs[0]
        elif ufunc is np.not_equal:
            return self != inputs[0]
        else:
            return NotImplemented

    setattr(cls, "__array_ufunc__", __array_ufunc__)
    return cls
