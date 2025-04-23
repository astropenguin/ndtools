__all__ = ["eq", "ge", "gt", "le", "lt", "ne"]


# standard library
from typing import Any, TypeVar


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
    if getattr(cls := type(left), "__eq__", eq) is not eq:
        return left == right

    if getattr(cls, "__ne__", ne) is not ne:
        return eq_by_ne(left, right)

    raise AttributeError("No comparison operator is defined for left == right.")


def eq_by_ne(left: T, right: Any, /) -> T:
    """Implement the ``==`` operator by ``not(!=)``."""
    return ~(left != right)


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
    if getattr(cls := type(left), "__ge__", ge) is not ge:
        return left >= right

    if getattr(cls, "__lt__", lt) is not lt:
        return ge_by_lt(left, right)

    if getattr(cls, "__gt__", gt) is not gt:
        return ge_by_gt(left, right)

    if getattr(cls, "__le__", le) is not le:
        return ge_by_le(left, right)

    raise AttributeError("No comparison operator is defined for left >= right.")


def ge_by_gt(left: T, right: Any, /) -> T:
    """Implement the ``>=`` operator by ``> or ==``."""
    return (left > right) | (left == right)


def ge_by_le(left: T, right: Any, /) -> T:
    """Implement the ``>=`` operator by ``not(<=) or ==``."""
    return ~(left <= right) | (left == right)


def ge_by_lt(left: T, right: Any, /) -> T:
    """Implement the ``>=`` operator by ``not(<)``."""
    return ~(left < right)


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
    if getattr(cls := type(left), "__gt__", gt) is not gt:
        return left > right

    if getattr(cls, "__le__", le) is not le:
        return gt_by_le(left, right)

    if getattr(cls, "__ge__", ge) is not ge:
        return gt_by_ge(left, right)

    if getattr(cls, "__lt__", lt) is not lt:
        return gt_by_lt(left, right)

    raise AttributeError("No comparison operator is defined for left > right.")


def gt_by_ge(left: T, right: Any, /) -> T:
    """Implement the ``>`` operator by ``>= and !=``."""
    return (left >= right) & (left != right)


def gt_by_le(left: T, right: Any, /) -> T:
    """Implement the ``>`` operator by ``not(<=)``."""
    return ~(left <= right)


def gt_by_lt(left: T, right: Any, /) -> T:
    """Implement the ``>`` operator by ``not(<) and !=``."""
    return ~(left < right) & (left != right)


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
    if getattr(cls := type(left), "__le__", le) is not le:
        return left <= right

    if getattr(cls, "__gt__", gt) is not gt:
        return le_by_gt(left, right)

    if getattr(cls, "__lt__", lt) is not lt:
        return le_by_lt(left, right)

    if getattr(cls, "__ge__", ge) is not ge:
        return le_by_ge(left, right)

    raise AttributeError("No comparison operator is defined for left <= right.")


def le_by_ge(left: T, right: Any, /) -> T:
    """Implement the ``<=`` operator by ``not(>=) or ==``."""
    return ~(left >= right) | (left == right)


def le_by_gt(left: T, right: Any, /) -> T:
    """Implement the ``<=`` operator by ``not(>)``."""
    return ~(left > right)


def le_by_lt(left: T, right: Any, /) -> T:
    """Implement the ``<=`` operator by ``< or ==``."""
    return (left < right) | (left == right)


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
    if getattr(cls := type(left), "__lt__", lt) is not lt:
        return left < right

    if getattr(cls, "__ge__", ge) is not ge:
        return lt_by_ge(left, right)

    if getattr(cls, "__le__", le) is not le:
        return lt_by_le(left, right)

    if getattr(cls, "__gt__", gt) is not gt:
        return lt_by_gt(left, right)

    raise AttributeError("No comparison operator is defined for left < right.")


def lt_by_ge(left: T, right: Any, /) -> T:
    """Implement the ``<`` operator by ``not(>=)``."""
    return ~(left >= right)


def lt_by_gt(left: T, right: Any, /) -> T:
    """Implement the ``<`` operator by ``not(>) and !=``."""
    return ~(left > right) & (left != right)


def lt_by_le(left: T, right: Any, /) -> T:
    """Implement the ``<`` operator by ``<= and !=``."""
    return (left <= right) & (left != right)


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
    if getattr(cls := type(left), "__ne__", ne) is not ne:
        return left != right

    if getattr(cls, "__eq__", eq) is not eq:
        return ne_by_eq(left, right)

    raise AttributeError("No comparison operator is defined for left != right.")


def ne_by_eq(left: T, right: Any, /) -> T:
    """Implement the ``!=`` operator by ``not(==)``."""
    return ~(left == right)
