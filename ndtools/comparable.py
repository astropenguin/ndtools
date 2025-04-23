__all__ = ["Equatable", "Orderable"]


# standard library
from abc import ABC, abstractmethod
from typing import Any


# dependencies
import numpy as np


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
    def __eq__(self, other: Any) -> Any:
        pass

    @abstractmethod
    def __ne__(self, other: Any) -> Any:
        pass

    def __array_ufunc__(
        self: Any,
        ufunc: np.ufunc,
        method: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
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
    def __eq__(self, other: Any) -> Any:
        pass

    @abstractmethod
    def __ge__(self, other: Any) -> Any:
        pass

    @abstractmethod
    def __gt__(self, other: Any) -> Any:
        pass

    @abstractmethod
    def __le__(self, other: Any) -> Any:
        pass

    @abstractmethod
    def __lt__(self, other: Any) -> Any:
        pass

    @abstractmethod
    def __ne__(self, other: Any) -> Any:
        pass

    def __array_ufunc__(
        self: Any,
        ufunc: np.ufunc,
        method: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
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
