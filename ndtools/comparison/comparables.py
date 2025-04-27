__all__ = ["All", "Any", "Combinable", "Equatable", "Not", "Orderable"]


# standard library
from collections import UserList
from collections.abc import Callable, Iterable
from functools import reduce
from operator import and_, or_
from typing import Any as Any_


# dependencies
import numpy as np
from .operators import eq, ge, gt, le, lt, ne
from .utils import has_method


class Combinable:
    """Implement logical operations between comparables.

    Classes that inherit from this mixin class can perform
    logical operations between comparables.
    Then ``comparable_0 & comparable_1 & ...``
    will return ``All([comparable_0, comparable_1, ...])``
    and ``comparable_0 | comparable_1 | ...``
    will return ``Any([comparable_0, comparable_1, ...])``.
    where ``All`` and ``Any`` are the implementation of
    logical conjunction and logical disjunction, respectively.

    Examples:
        ::

            import numpy as np
            from ndtools import Combinable, Equatable

            class Even(Combinable, Equatable):
                def __eq__(self, array):
                    return array % 2 == 0

                def __ne__(self, array):
                    return ~(self == array)

            class Odd(Combinable, Equatable):
                def __eq__(self, array):
                    return array % 2 == 1

                def __ne__(self, array):
                    return ~(self == array)

            Even() & Odd()  # -> All([Even(), Odd()])
            Even() | Odd()  # -> Any([Even(), Odd()])

            np.arange(3) == Even() & Odd()  # -> array([False, False, False])
            np.arange(3) == Even() | Odd()  # -> array([True, True, True])

    """

    def __and__(self, other: Any_) -> "All":
        def iterable(obj: Any_) -> Iterable[Any_]:
            return obj if isinstance(obj, All) else [obj]

        return All([*iterable(self), *iterable(other)])

    def __or__(self, other: Any_) -> "Any":
        def iterable(obj: Any_) -> Iterable[Any_]:
            return obj if isinstance(obj, Any) else [obj]

        return Any([*iterable(self), *iterable(other)])


class Equatable:
    """Implement equality operations for multidimensional arrays.

    Classes that inherit from this mixin class
    and implement ``__eq__`` or ``__ne__`` special methods
    can perform their own equality operations on multidimensional arrays.
    These special methods should be implemented for the target array like
    ``def __eq__(self, array)``. Then the class instance and the array
    can perform ``instance == array`` and ``array == instance``.

    Examples:
        ::

            import numpy as np
            from ndtools import Equatable

            class Even(Equatable):
                def __eq__(self, array):
                    return array % 2 == 0

            Even() == np.arange(3)  # -> array([True, False, True])
            np.arange(3) == Even()  # -> array([True, False, True])

            Even() != np.arange(3)  # -> array([False, True, False])
            np.arange(3) != Even()  # -> array([False, True, False])

    """

    __eq__: Callable[..., Any_]
    __ne__: Callable[..., Any_]

    def __array_ufunc__(
        self: Any_,
        ufunc: np.ufunc,
        method: str,
        *inputs: Any_,
        **kwargs: Any_,
    ) -> Any_:
        if ufunc is np.equal:
            return self == inputs[0]

        if ufunc is np.not_equal:
            return self != inputs[0]

        return NotImplemented

    def __init_subclass__(cls, **kwargs: Any_) -> None:
        super().__init_subclass__(**kwargs)

        for operator in (eq, ne):
            if not has_method(cls, f"__{operator.__name__}__"):
                setattr(cls, f"__{operator.__name__}__", operator)


class Orderable:
    """Implement ordering operations for multidimensional arrays.

    Classes that inherit from this mixin base class
    and implement both (1) ``__eq__`` or ``__ne__`` special methods
    and (2) ``__ge__``, ``__gt__``, ``__le__``, or ``__lt__`` special methods
    can perform their own ordering operations on multidimensional arrays.
    These special methods should be implemented for the target array like
    ``def __ge__(self, array)``. Then the class instance and the array
    can perform ``instance >= array`` and ``array <= instance``.

    Examples:
        ::

            import numpy as np
            from dataclasses import dataclass
            from ndtools import Orderable

            @dataclass
            class Range(Orderable):
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

    __eq__: Callable[..., Any_]
    __ge__: Callable[..., Any_]
    __gt__: Callable[..., Any_]
    __le__: Callable[..., Any_]
    __lt__: Callable[..., Any_]
    __ne__: Callable[..., Any_]

    def __array_ufunc__(
        self: Any_,
        ufunc: np.ufunc,
        method: str,
        *inputs: Any_,
        **kwargs: Any_,
    ) -> Any_:
        if ufunc is np.equal:
            return self == inputs[0]

        if ufunc is np.greater:
            return self < inputs[0]

        if ufunc is np.greater_equal:
            return self <= inputs[0]

        if ufunc is np.less:
            return self > inputs[0]

        if ufunc is np.less_equal:
            return self >= inputs[0]

        if ufunc is np.not_equal:
            return self != inputs[0]

        return NotImplemented

    def __init_subclass__(cls, **kwargs: Any_) -> None:
        super().__init_subclass__(**kwargs)

        for operator in (eq, ge, gt, le, lt, ne):
            if not has_method(cls, f"__{operator.__name__}__"):
                setattr(cls, f"__{operator.__name__}__", operator)


class All(UserList[Any_], Combinable, Equatable):
    """Implement logical conjunction between comparables.

    It should contain comparables like ``All([comparable_0, comparable_1, ...])``.
    Then the equality operation on the target array will perform like
    ``(array == comparable_0) & array == comparable_1) & ...``.

    Examples:
        ::

            import numpy as np
            from ndtools import Combinable, Equatable

            class Even(Combinable, Equatable):
                def __eq__(self, array):
                    return array % 2 == 0

                def __ne__(self, array):
                    return ~(self == array)

            class Odd(Combinable, Equatable):
                def __eq__(self, array):
                    return array % 2 == 1

                def __ne__(self, array):
                    return ~(self == array)

            Even() & Odd()  # -> All([Even(), Odd()])
            np.arange(3) == Even() & Odd()  # -> array([False, False, False])

    """

    def __eq__(self, other: Any_) -> Any_:
        return reduce(and_, (other == cond for cond in self))


class Any(UserList[Any_], Combinable, Equatable):
    """Implement logical disjunction between comparables.

    It should contain comparables like ``Any([comparable_0, comparable_1, ...])``.
    Then the equality operation on the target array will perform like
    ``(array == comparable_0) | array == comparable_1) & ...``.

    Examples:
        ::

            import numpy as np
            from ndtools import Combinable, Equatable

            class Even(Combinable, Equatable):
                def __eq__(self, array):
                    return array % 2 == 0

                def __ne__(self, array):
                    return ~(self == array)

            class Odd(Combinable, Equatable):
                def __eq__(self, array):
                    return array % 2 == 1

                def __ne__(self, array):
                    return ~(self == array)

            Even() | Odd()  # -> Any([Even(), Odd()])
            np.arange(3) == Even() | Odd()  # -> array([True, True, True])

    """

    def __eq__(self, other: Any_) -> Any_:
        return reduce(or_, (other == cond for cond in self))


class Not(Combinable, Equatable):
    """Implement logical negation for comparables.

    It should wrap a comparable like ``Not(comparable)``.
    Then the equality operation on the target array
    will perform like ``array != comparable``.

    Examples:
        ::

            import numpy as np
            from ndtools import Not

            np.arange(3) == Not(1)  # -> array([True, False, True])

    """

    def __init__(self, comparable: Any_, /) -> None:
        self.comparable = comparable

    def __eq__(self, other: Any_) -> Any_:
        return other != self.comparable

    def __repr__(self) -> str:
        return f"Not({self.comparable})"
