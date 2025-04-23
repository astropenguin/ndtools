__all__ = [
    "All",
    "Any",
    "Combinable",
    "Equatable",
    "Orderable",
    "TotalEquality",
    "TotalOrdering",
]


# standard library
from abc import ABC, abstractmethod
from collections import UserList
from collections.abc import Callable, Iterable
from functools import reduce
from operator import and_, or_
from typing import Any as Any_


# dependencies
import numpy as np
from . import operators as op


class Combinable:
    """Implement logical operations between objects.

    Classes that inherit from this mix-in class can perform logical
    operations between the class instance and other object.
    Then ``instance & object`` will return ``All([instance, other])``
    and ``instance | object`` will return ``Any[instance, other])``,
    where ``All`` and ``Any`` are the implementation of
    logical conjunction and logical disjunction, respectively.
    In general, ``Combinable`` should be used with the ``Equatable``
    abstract base class to implement combinable equatables.

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


class Equatable(ABC):
    """Implement equality operations for multidimensional arrays.

    Classes that inherit from this abstract base class
    and implement both ``__eq__`` and ``__ne__`` special methods
    can perform their own equality operations on multidimensional arrays.
    These special methods should be implemented for the target array like
    ``def __eq__(self, array)``. Then the class instance and the array
    can perform ``instance == array`` and ``array == instance``.

    Raises:
        TypeError: Raised if both ``__eq__`` and ``__ne__`` are not defined.

    Examples:
        ::

            import numpy as np
            from ndtools import Equatable

            class Even(Equatable):
                def __eq__(self, array):
                    return array % 2 == 0

                def __ne__(self, array):
                    return ~(self == array)

            Even() == np.arange(3)  # -> array([True, False, True])
            np.arange(3) == Even()  # -> array([True, False, True])

            Even() != np.arange(3)  # -> array([False, True, False])
            np.arange(3) != Even()  # -> array([False, True, False])

    """

    @abstractmethod
    def __eq__(self, other: Any_) -> Any_:
        pass

    @abstractmethod
    def __ne__(self, other: Any_) -> Any_:
        pass

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


class Orderable(ABC):
    """Implement ordering operations for multidimensional arrays.

    Classes that inherit from this abstract base class
    and implement all of ``__eq__``, ``__ge__``, ``__gt__``,
    ``__le__``, ``__lt__``, and ``__ne__`` special methods
    can perform their own ordering operations on multidimensional arrays.
    These special methods should be implemented for the target array like
    ``def __ge__(self, array)``. Then the class instance and the array
    can perform ``instance >= array`` and ``array <= instance``.

    Raises:
        TypeError: Raised if all of ``__eq__``, ``__ge__``, ``__gt__``,
            ``__le__``, ``__lt__``, and ``__ne__`` are not defined.

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

                def __gt__(self, array):
                    return array < self.lower

                def __le__(self, array):
                    return ~(self > array)

                def __lt__(self, array):
                    return ~(self >= array)

                def __ne__(self, array):
                    return ~(self == array)

            Range(1, 2) == np.arange(3)  # -> array([False, True, False])
            np.arange(3) == Range(1, 2)  # -> array([False, True, False])

            Range(1, 2) >= np.arange(3)  # -> array([True, True, False])
            np.arange(3) <= Range(1, 2)  # -> array([True, True, False])

    """

    @abstractmethod
    def __eq__(self, other: Any_) -> Any_:
        pass

    @abstractmethod
    def __ge__(self, other: Any_) -> Any_:
        pass

    @abstractmethod
    def __gt__(self, other: Any_) -> Any_:
        pass

    @abstractmethod
    def __le__(self, other: Any_) -> Any_:
        pass

    @abstractmethod
    def __lt__(self, other: Any_) -> Any_:
        pass

    @abstractmethod
    def __ne__(self, other: Any_) -> Any_:
        pass

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


class All(UserList[Any_], Combinable, Equatable):
    """Implement logical conjunction between equatables.

    It should contain equatables like ``All([eqatable_0, equatable_1, ...])``.
    Then the equality operation on the target array will perform like
    ``(array == equatable_0) & array == equatable_1) & ...``.

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

    def __ne__(self, other: Any_) -> Any_:
        return ~(self == other)


class Any(UserList[Any_], Combinable, Equatable):
    """Implement logical disjunction between equatables.

    It should contain equatables like ``Any([eqatable_0, equatable_1, ...])``.
    Then the equality operation on the target array will perform like
    ``(array == equatable_0) | array == equatable_1) & ...``.

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

    def __ne__(self, other: Any_) -> Any_:
        return ~(self == other)


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

    __eq__: Callable[..., Any_]
    __ne__: Callable[..., Any_]
    __array_ufunc__: Callable[..., Any_]

    def __init_subclass__(cls, **kwargs: Any_) -> None:
        super().__init_subclass__(**kwargs)

        for name in ("eq", "ne"):
            if not has_usermethod(cls, f"__{name}__"):
                setattr(cls, f"__{name}__", getattr(op, name))


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

    __eq__: Callable[..., Any_]
    __ge__: Callable[..., Any_]
    __gt__: Callable[..., Any_]
    __le__: Callable[..., Any_]
    __lt__: Callable[..., Any_]
    __ne__: Callable[..., Any_]
    __array_ufunc__: Callable[..., Any_]

    def __init_subclass__(cls, **kwargs: Any_) -> None:
        super().__init_subclass__(**kwargs)

        for name in ("eq", "ge", "gt", "le", "lt", "ne"):
            if not has_usermethod(cls, f"__{name}__"):
                setattr(cls, f"__{name}__", getattr(op, name))


def has_usermethod(obj: Any_, name: str, /) -> bool:
    """Check if an object has a user-defined method with given name."""
    return (
        hasattr(obj, name)
        and not is_abstractmethod(getattr(obj, name))
        and not is_objectmethod(getattr(obj, name))
    )


def is_abstractmethod(method: Any_, /) -> bool:
    """Check if given method is an abstract method."""
    return bool(getattr(method, "__isabstractmethod__", None))


def is_objectmethod(method: Any_, /) -> bool:
    """Check if given method is defined in the object class."""
    return method is getattr(object, method.__name__, None)
