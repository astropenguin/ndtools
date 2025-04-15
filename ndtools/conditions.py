__all__ = ["All", "Any"]


# standard library
from collections import UserList
from functools import reduce
from operator import and_, or_
from typing import Any as Any_


# dependencies
from typing_extensions import Self
from .comparison import total_equality


@total_equality
class All(UserList[Any_]):
    """"""

    def __and__(self, other: Any_) -> Self:
        if isinstance(other, type(self)):
            return self + other
        else:
            return self + [other]

    def __eq__(self, other: Any_) -> Any_:
        return reduce(and_, (other == cond for cond in self))

    def __or__(self, other: Any_) -> "Any":
        return Any([self, other])


@total_equality
class Any(UserList[Any_]):
    """"""

    def __and__(self, other: Any_) -> "All":
        return All([self, other])

    def __eq__(self, other: Any_) -> Any_:
        return reduce(or_, (other == cond for cond in self))

    def __or__(self, other: Any_) -> Self:
        if isinstance(other, type(self)):
            return self + other
        else:
            return self + [other]
