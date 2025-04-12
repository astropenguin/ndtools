__all__ = []


# standard library
from typing import Any, TypeVar


# type hints
T = TypeVar("T")


def eq_by_ne(left: T, right: Any, /) -> T:
    """Implement == by not(!=)."""
    return ~(left != right)


def ge_by_gt(left: T, right: Any, /) -> T:
    """Implement >= by (> or ==)."""
    return (left > right) | (left == right)


def ge_by_le(left: T, right: Any, /) -> T:
    """Implement >= by (not(<=) or ==)."""
    return ~(left <= right) | (left == right)


def ge_by_lt(left: T, right: Any, /) -> T:
    """Implement >= by not(<)."""
    return ~(left < right)


def gt_by_ge(left: T, right: Any, /) -> T:
    """Implement > by (>= and !=)."""
    return (left >= right) & (left != right)


def gt_by_le(left: T, right: Any, /) -> T:
    """Implement > by not(<=)."""
    return ~(left <= right)


def gt_by_lt(left: T, right: Any, /) -> T:
    """Implement > by (not(<) & !=)."""
    return ~(left < right) & (left != right)


def le_by_ge(left: T, right: Any, /) -> T:
    """Implement <= by (not(>=) or ==)."""
    return ~(left >= right) | (left == right)


def le_by_gt(left: T, right: Any, /) -> T:
    """Implement <= by not(>)."""
    return ~(left > right)


def le_by_lt(left: T, right: Any, /) -> T:
    """Implement <= by (< or ==)."""
    return (left < right) | (left == right)


def lt_by_ge(left: T, right: Any, /) -> T:
    """Implement < by not(>=)."""
    return ~(left >= right)


def lt_by_gt(left: T, right: Any, /) -> T:
    """Implement < by (not(>) and !=)."""
    return ~(left > right) & (left != right)


def lt_by_le(left: T, right: Any, /) -> T:
    """Implement < by (<= and !=)."""
    return (left <= right) & (left != right)


def ne_by_eq(left: T, right: Any, /) -> T:
    """Implement != by not(==)."""
    return ~(left == right)
