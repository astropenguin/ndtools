__all__ = [
    "eq_by_ne",
    "ge_by_gt",
    "ge_by_le",
    "ge_by_lt",
    "gt_by_ge",
    "gt_by_le",
    "gt_by_lt",
    "le_by_ge",
    "le_by_gt",
    "le_by_lt",
    "lt_by_ge",
    "lt_by_gt",
    "lt_by_le",
    "ne_by_eq",
]


# standard library
from typing import Any, TypeVar


# type hints
T = TypeVar("T")


def eq_by_ne(left: T, right: Any, /) -> T:
    """Implement the ``==`` operator by ``not(!=)``.

    Args:
        left: Left hand side of the operator.
        right: Right hand side of the operator.

    Returns:
        Result of ``left == right``.

    """
    return ~(left != right)


def ge_by_gt(left: T, right: Any, /) -> T:
    """Implement the ``>=`` operator by ``> or ==``.

    Args:
        left: Left hand side of the operator.
        right: Right hand side of the operator.

    Returns:
        Result of ``left >= right``.

    """
    return (left > right) | (left == right)


def ge_by_le(left: T, right: Any, /) -> T:
    """Implement the ``>=`` operator by ``not(<=) or ==``.

    Args:
        left: Left hand side of the operator.
        right: Right hand side of the operator.

    Returns:
        Result of ``left >= right``.

    """
    return ~(left <= right) | (left == right)


def ge_by_lt(left: T, right: Any, /) -> T:
    """Implement the ``>=`` operator by ``not(<)``.

    Args:
        left: Left hand side of the operator.
        right: Right hand side of the operator.

    Returns:
        Result of ``left >= right``.

    """
    return ~(left < right)


def gt_by_ge(left: T, right: Any, /) -> T:
    """Implement the ``>`` operator by ``>= and !=``.

    Args:
        left: Left hand side of the operator.
        right: Right hand side of the operator.

    Returns:
        Result of ``left > right``.

    """
    return (left >= right) & (left != right)


def gt_by_le(left: T, right: Any, /) -> T:
    """Implement the ``>`` operator by ``not(<=)``.

    Args:
        left: Left hand side of the operator.
        right: Right hand side of the operator.

    Returns:
        Result of ``left > right``.

    """
    return ~(left <= right)


def gt_by_lt(left: T, right: Any, /) -> T:
    """Implement the ``>`` operator by ``not(<) and !=``.

    Args:
        left: Left hand side of the operator.
        right: Right hand side of the operator.

    Returns:
        Result of ``left > right``.

    """
    return ~(left < right) & (left != right)


def le_by_ge(left: T, right: Any, /) -> T:
    """Implement the ``<=`` operator by ``not(>=) or ==``.

    Args:
        left: Left hand side of the operator.
        right: Right hand side of the operator.

    Returns:
        Result of ``left <= right``.

    """
    return ~(left >= right) | (left == right)


def le_by_gt(left: T, right: Any, /) -> T:
    """Implement the ``<=`` operator by ``not(>)``.

    Args:
        left: Left hand side of the operator.
        right: Right hand side of the operator.

    Returns:
        Result of ``left <= right``.

    """
    return ~(left > right)


def le_by_lt(left: T, right: Any, /) -> T:
    """Implement the ``<=`` operator by ``< or ==``.

    Args:
        left: Left hand side of the operator.
        right: Right hand side of the operator.

    Returns:
        Result of ``left <= right``.

    """
    return (left < right) | (left == right)


def lt_by_ge(left: T, right: Any, /) -> T:
    """Implement the ``<`` operator by ``not(>=)``.

    Args:
        left: Left hand side of the operator.
        right: Right hand side of the operator.

    Returns:
        Result of ``left < right``.

    """
    return ~(left >= right)


def lt_by_gt(left: T, right: Any, /) -> T:
    """Implement the ``<`` operator by ``not(>) and !=``.

    Args:
        left: Left hand side of the operator.
        right: Right hand side of the operator.

    Returns:
        Result of ``left < right``.

    """
    return ~(left > right) & (left != right)


def lt_by_le(left: T, right: Any, /) -> T:
    """Implement the ``<`` operator by ``<= and !=``.

    Args:
        left: Left hand side of the operator.
        right: Right hand side of the operator.

    Returns:
        Result of ``left < right``.

    """
    return (left <= right) & (left != right)


def ne_by_eq(left: T, right: Any, /) -> T:
    """Implement the ``!=`` operator by ``not(==)``.

    Args:
        left: Left hand side of the operator.
        right: Right hand side of the operator.

    Returns:
        Result of ``left != right``.

    """
    return ~(left == right)
