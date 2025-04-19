__all__ = ["TotalEquality", "TotalOrdering"]


# standard library
from collections.abc import Callable
from typing import Any, TypeVar


# dependencies
from . import operators as op
from .abc import Equatable, Orderable


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


class TotalEquality(Equatable):
    """Implement missing equality operations for multidimensional arrays.

    Raises:
        ValueError: Raised if none of the equality operators (==, !=) is defined.

    Examples:
        ::

            import numpy as np
            from ndtools import TotalEquality

            class Even(TotalEquality):
                def __eq__(self, array):
                    return array % 2 == 0

            Even() == np.arange(3)  # -> array([True, False, True])
            np.arange(3) == Even()  # -> array([True, False, True])

            Even() != np.arange(3)  # -> array([False, True, False])
            np.arange(3) != Even()  # -> array([False, True, False])

    """

    __eq__: Callable[..., Any]
    __ne__: Callable[..., Any]
    __array_ufunc__: Callable[..., Any]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        defined = [name for name in MISSINGS_EQUALITY if has_usermethod(cls, name)]

        if not defined:
            raise ValueError("Define at least one equality operator (==, !=).")

        for name, operator in MISSINGS_EQUALITY[defined[0]].items():
            if not has_usermethod(cls, name):
                setattr(cls, name, operator)


class TotalOrdering(Orderable):
    """Implement missing ordering operations for multidimensional arrays.

    Raises:
        ValueError: Raise if none of the ordering operator (>=, >, <=, <) is defined.

    Examples:
        ::

            import numpy as np
            from dataclasses import dataclass
            from ndtools import TotalOrdering

            @dataclass
            class Range(TotalOrdering):
                lower: float
                upper: float

                def __eq__(self, array):
                    return (array >= self.lower) & (array < self.upper)

                def __ge__(self, array):
                    return array < self.upper

            Range(1, 2) == np.arange(3)  # -> array([False, True, False])
            np.arange(3) == Range(1, 2)  # -> array([False, True, False])

            Range(1, 2) >= np.arange(3)  # -> array([True, True, False])
            np.arange(3) <= Range(1, 2)  # -> array([True, True, False])

    """

    __eq__: Callable[..., Any]
    __ge__: Callable[..., Any]
    __gt__: Callable[..., Any]
    __le__: Callable[..., Any]
    __lt__: Callable[..., Any]
    __ne__: Callable[..., Any]
    __array_ufunc__: Callable[..., Any]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        defined = [name for name in MISSINGS_EQUALITY if has_usermethod(cls, name)]

        if not defined:
            raise ValueError("Define at least one equality operator (==, !=).")

        for name, operator in MISSINGS_EQUALITY[defined[0]].items():
            if not has_usermethod(cls, name):
                setattr(cls, name, operator)

        defined = [name for name in MISSINGS_ORDERING if has_usermethod(cls, name)]

        if not defined:
            raise ValueError("Define at least one ordering operator (>=, >, <=, <).")

        for name, operator in MISSINGS_ORDERING[defined[0]].items():
            if not has_usermethod(cls, name):
                setattr(cls, name, operator)


def has_usermethod(obj: Any, name: str, /) -> bool:
    """Check if an object has a user-defined method with given name."""
    return (
        hasattr(obj, name)
        and not is_abstractmethod(getattr(obj, name))
        and not is_objectmethod(getattr(obj, name))
    )


def is_abstractmethod(method: Any, /) -> bool:
    """Check if given method is an abstract method."""
    return bool(getattr(method, "__isabstractmethod__", None))


def is_objectmethod(method: Any, /) -> bool:
    """Check if given method is defined in the object class."""
    return method is getattr(object, method.__name__, None)
