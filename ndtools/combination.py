__all__ = ["All", "Any", "Combinable"]


# standard library
from collections import UserList
from collections.abc import Iterable
from functools import reduce
from operator import and_, or_
from typing import Any as Any_


# dependencies
from .abc import Equatable


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
