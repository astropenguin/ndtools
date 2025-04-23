__all__ = ["eq", "ge", "gt", "le", "lt", "ne"]


# standard library
from typing import Any, TypeVar


# dependencies
from .utils import get_method


# type hints
T = TypeVar("T")


def eq(left: T, right: Any, /) -> T:
    """Implement the ``==`` operator for multidimensional arrays.

    If ``left`` does not implement the ``__eq__`` method, it will fall back
    to an equivalent implementation using another comparison operators.

    Args:
        left: Left hand side of the operator.
        right: Right hand side of the operator.

    Returns:
        Result of ``left == right``.

    Raises:
        AttributeError: Raised if no comparison operator is defined for ``left == right``.

    """
    if get_method(cls := type(left), "__eq__", eq) is not eq:
        return left == right

    if get_method(cls, "__ne__", ne) is not ne:
        return ~ne(left, right)  # type: ignore

    raise AttributeError("No comparison operator is defined for left == right.")


def ge(left: T, right: Any, /) -> T:
    """Implement the ``>=`` operator for multidimensional arrays.

    If ``left`` does not implement the ``__ge__`` method, it will fall back
    to an equivalent implementation using another comparison operators.

    Args:
        left: Left hand side of the operator.
        right: Right hand side of the operator.

    Returns:
        Result of ``left >= right``.

    Raises:
        AttributeError: Raised if no comparison operator is defined for ``left >= right``.

    """
    if get_method(cls := type(left), "__ge__", ge) is not ge:
        return left >= right

    if get_method(cls, "__lt__", lt) is not lt:
        return ~lt(left, right)  # type: ignore

    if get_method(cls, "__gt__", gt) is not gt:
        return gt(left, right) | eq(left, right)  # type: ignore

    if get_method(cls, "__le__", le) is not le:
        return ~le(left, right) | eq(left, right)  # type: ignore

    raise AttributeError("No comparison operator is defined for left >= right.")


def gt(left: T, right: Any, /) -> T:
    """Implement the ``>`` operator for multidimensional arrays.

    If ``left`` does not implement the ``__gt__`` method, it will fall back
    to an equivalent implementation using another comparison operators.

    Args:
        left: Left hand side of the operator.
        right: Right hand side of the operator.

    Returns:
        Result of ``left > right``.

    Raises:
        AttributeError: Raised if no comparison operator is defined for ``left > right``.

    """
    if get_method(cls := type(left), "__gt__", gt) is not gt:
        return left > right

    if get_method(cls, "__le__", le) is not le:
        return ~le(left, right)  # type: ignore

    if get_method(cls, "__ge__", ge) is not ge:
        return ge(left, right) & ne(left, right)  # type: ignore

    if get_method(cls, "__lt__", lt) is not lt:
        return ~lt(left, right) & ne(left, right)  # type: ignore

    raise AttributeError("No comparison operator is defined for left > right.")


def le(left: T, right: Any, /) -> T:
    """Implement the ``<=`` operator for multidimensional arrays.

    If ``left`` does not implement the ``__le__`` method, it will fall back
    to an equivalent implementation using another comparison operators.

    Args:
        left: Left hand side of the operator.
        right: Right hand side of the operator.

    Returns:
        Result of ``left <= right``.

    Raises:
        AttributeError: Raised if no comparison operator is defined for ``left <= right``.

    """
    if get_method(cls := type(left), "__le__", le) is not le:
        return left <= right

    if get_method(cls, "__gt__", gt) is not gt:
        return ~gt(left, right)  # type: ignore

    if get_method(cls, "__lt__", lt) is not lt:
        return lt(left, right) | eq(left, right)  # type: ignore

    if get_method(cls, "__ge__", ge) is not ge:
        return ~ge(left, right) | eq(left, right)  # type: ignore

    raise AttributeError("No comparison operator is defined for left <= right.")


def lt(left: T, right: Any, /) -> T:
    """Implement the ``<`` operator for multidimensional arrays.

    If ``left`` does not implement the ``__lt__`` method, it will fall back
    to an equivalent implementation using another comparison operators.

    Args:
        left: Left hand side of the operator.
        right: Right hand side of the operator.

    Returns:
        Result of ``left < right``.

    Raises:
        AttributeError: Raised if no comparison operator is defined for ``left < right``.

    """
    if get_method(cls := type(left), "__lt__", lt) is not lt:
        return left < right

    if get_method(cls, "__ge__", ge) is not ge:
        return ~ge(left, right)  # type: ignore

    if get_method(cls, "__le__", le) is not le:
        return le(left, right) & ne(left, right)  # type: ignore

    if get_method(cls, "__gt__", gt) is not gt:
        return ~gt(left, right) & ne(left, right)  # type: ignore

    raise AttributeError("No comparison operator is defined for left < right.")


def ne(left: T, right: Any, /) -> T:
    """Implement the ``!=`` operator for multidimensional arrays.

    If ``left`` does not implement the ``__ne__`` method, it will fall back
    to an equivalent implementation using another comparison operators.

    Args:
        left: Left hand side of the operator.
        right: Right hand side of the operator.

    Returns:
        Result of ``left != right``.

    Raises:
        AttributeError: Raised if no comparison operator is defined for ``left != right``.

    """
    if get_method(cls := type(left), "__ne__", ne) is not ne:
        return left != right

    if get_method(cls, "__eq__", eq) is not eq:
        return ~eq(left, right)  # type: ignore

    raise AttributeError("No comparison operator is defined for left != right.")
