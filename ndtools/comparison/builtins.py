__all__ = ["ANY", "NEVER", "AnyType", "NeverType"]


# standard library
from typing import Any as Any_


# dependencies
from typing_extensions import Self
from .comparables import Combinable, Equatable


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
