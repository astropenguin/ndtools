__all__ = [
    "first",
    "nanfirst",
    "last",
    "nanlast",
    "mad",
    "nanmad",
    "middle",
    "nanmiddle",
]

# standard library
from collections.abc import Callable, Sequence
from typing import Any, Literal

# dependencies
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

# constants
FIRST_AXIS = 0
FIRST_INDEX = 0
LAST_AXIS = -1
LAST_INDEX = -1
NEW_AXIS = None


def first(
    array: ArrayLike,
    /,
    *,
    axis: Sequence[int] | int | None = None,
    keepdims: bool = False,
) -> NDArray[Any] | Any:
    """Compute the first element along the specified axis.

    Args:
        array: Input array or object that can be converted to an array.
        axis: Axis or axes along which the first element is computed.
        keepdims: Whether to retain the reduced axes as dimensions with size one.

    Returns:
        A new array (or scalar) holding the computed first element.
    """

    def func(array: NDArray[Any], /) -> NDArray[Any]:
        indices = np.full(array.shape[:LAST_AXIS], FIRST_INDEX)[..., NEW_AXIS]
        return np.take_along_axis(array, indices, LAST_AXIS).squeeze(LAST_AXIS)

    return _reduce(array, func, axis=axis, keepdims=keepdims)


def nanfirst(
    array: ArrayLike,
    /,
    *,
    axis: Sequence[int] | int | None = None,
    keepdims: bool = False,
) -> NDArray[Any] | Any:
    """Compute the first element along the specified axis, ignoring NaNs.

    If the exact first element is NaN, the function searches towards the
    end (right side) of the array along the given axis until the first
    non-NaN element is found. If all elements along the specified axis
    are NaN, the result will be NaN.

    Args:
        array: Input array or object that can be converted to an array.
        axis: Axis or axes along which the first non-NaN element is computed.
        keepdims: Whether to retain the reduced axes as dimensions with size one.

    Returns:
        A new array (or scalar) holding the computed first element.
    """

    def func(array: NDArray[Any], /) -> NDArray[Any]:
        indices = np.argmax(~pd.isna(array), axis=LAST_AXIS)[..., NEW_AXIS]
        return np.take_along_axis(array, indices, LAST_AXIS).squeeze(LAST_AXIS)

    return _reduce(array, func, axis=axis, keepdims=keepdims)


def last(
    array: ArrayLike,
    /,
    *,
    axis: Sequence[int] | int | None = None,
    keepdims: bool = False,
) -> NDArray[Any] | Any:
    """Compute the last element along the specified axis.

    Args:
        array: Input array or object that can be converted to an array.
        axis: Axis or axes along which the last element is computed.
        keepdims: Whether to retain the reduced axes as dimensions with size one.

    Returns:
        A new array (or scalar) holding the computed last element.
    """

    def func(array: NDArray[Any], /) -> NDArray[Any]:
        indices = np.full(array.shape[:LAST_AXIS], LAST_INDEX)[..., NEW_AXIS]
        return np.take_along_axis(array, indices, LAST_AXIS).squeeze(LAST_AXIS)

    return _reduce(array, func, axis=axis, keepdims=keepdims)


def nanlast(
    array: ArrayLike,
    /,
    *,
    axis: Sequence[int] | int | None = None,
    keepdims: bool = False,
) -> NDArray[Any] | Any:
    """Compute the last element along the specified axis, ignoring NaNs.

    If the exact last element is NaN, the function searches towards the
    beginning (left side) of the array along the given axis until the first
    non-NaN element is found. If all elements along the specified axis
    are NaN, the result will be NaN.

    Args:
        array: Input array or object that can be converted to an array.
        axis: Axis or axes along which the last non-NaN element is computed.
        keepdims: Whether to retain the reduced axes as dimensions with size one.

    Returns:
        A new array (or scalar) holding the computed last element.
    """

    def func(array: NDArray[Any], /) -> NDArray[Any]:
        indices = (
            # fmt: off
            - np.argmax(~pd.isna(array)[..., ::-1], LAST_AXIS)[..., NEW_AXIS]
            + (array.shape[LAST_AXIS] - 1)
            # fmt: on
        )
        return np.take_along_axis(array, indices, LAST_AXIS).squeeze(LAST_AXIS)

    return _reduce(array, func, axis=axis, keepdims=keepdims)


def mad(
    array: ArrayLike,
    /,
    *,
    axis: Sequence[int] | int | None = None,
    keepdims: bool = False,
) -> NDArray[Any] | Any:
    """Compute the median absolute deviation along the specified axis.

    Args:
        array: Input array or object that can be converted to an array.
        axis: Axis or axes along which the MAD is computed.
        keepdims: Whether to retain the reduced axes as dimensions with size one.

    Returns:
        A new array (or scalar) holding the computed median absolute deviation.
    """
    array = np.asarray(array)
    absdev = np.abs(array - np.median(array, axis=axis, keepdims=True))
    return np.median(absdev, axis=axis, keepdims=keepdims)


def nanmad(
    array: ArrayLike,
    /,
    *,
    axis: Sequence[int] | int | None = None,
    keepdims: bool = False,
) -> NDArray[Any] | Any:
    """Compute the median absolute deviation along the specified axis, ignoring NaNs.

    Args:
        array: Input array or object that can be converted to an array.
        axis: Axis or axes along which the MAD (ignoreing NaNs) is computed.
        keepdims: Whether to retain the reduced axes as dimensions with size one.

    Returns:
        A new array (or scalar) holding the computed median absolute deviation.
    """
    array = np.asarray(array)
    absdev = np.abs(array - np.nanmedian(array, axis=axis, keepdims=True))
    return np.nanmedian(absdev, axis=axis, keepdims=keepdims)


def middle(
    array: ArrayLike,
    /,
    *,
    axis: Sequence[int] | int | None = None,
    keepdims: bool = False,
) -> NDArray[Any] | Any:
    """Compute the middle element along the specified axis.

    If the length of the target axis is even, the left-middle element is
    returned (e.g., the element at index 1 for an axis of length 4).

    Args:
        array: Input array or object that can be converted to an array.
        axis: Axis or axes along which the middle element is computed.
        keepdims: Whether to retain the reduced axes as dimensions with size one.

    Returns:
        A new array (or scalar) holding the computed middle element.
    """

    def func(array: NDArray[Any], /) -> NDArray[Any]:
        middle_index = (array.shape[LAST_AXIS] - 1) // 2
        indices = np.full(array.shape[:LAST_AXIS], middle_index)[..., NEW_AXIS]
        return np.take_along_axis(array, indices, LAST_AXIS).squeeze(LAST_AXIS)

    return _reduce(array, func, axis=axis, keepdims=keepdims)


def nanmiddle(
    array: ArrayLike,
    /,
    *,
    axis: Sequence[int] | int | None = None,
    side: Literal["left", "right"] = "left",
    keepdims: bool = False,
) -> NDArray[Any] | Any:
    """Compute the middle element along the specified axis, ignoring NaNs.

    If the length of the target axis is even, the left-middle element is
    returned (e.g., the element at index 1 for an axis of length 4).
    If this element is NaN, the function searches for the nearest non-NaN element.
    The ``side`` parameter controls the direction of this search.
    If ``side='left'``, the search proceeds towards the beginning of the array.
    If ``side='right'``, the search proceeds towards the end of the array.

    Args:
        array: Input array or object that can be converted to an array.
        axis: Axis or axes along which the middle non-NaN element is computed.
        side: The direction to search for non-NaN elements.
        keepdims: Whether to retain the reduced axes as dimensions with size one.

    Returns:
        A new array (or scalar) holding the computed middle element.
    """

    def func_left(array: NDArray[Any], /) -> NDArray[Any]:
        array = array[..., : (array.shape[LAST_AXIS] - 1) // 2 + 1]
        indices = (
            # fmt: off
            - np.argmax(~pd.isna(array)[..., ::-1], LAST_AXIS)[..., NEW_AXIS]
            + (array.shape[LAST_AXIS] - 1)
            # fmt: on
        )
        return np.take_along_axis(array, indices, LAST_AXIS).squeeze(LAST_AXIS)

    def func_right(array: NDArray[Any], /) -> NDArray[Any]:
        array = array[..., (array.shape[LAST_AXIS] - 1) // 2 :]
        indices = np.argmax(~pd.isna(array), axis=LAST_AXIS)[..., NEW_AXIS]
        return np.take_along_axis(array, indices, LAST_AXIS).squeeze(LAST_AXIS)

    if side == "left":
        return _reduce(array, func_left, axis=axis, keepdims=keepdims)
    elif side == "right":
        return _reduce(array, func_right, axis=axis, keepdims=keepdims)
    else:
        raise ValueError(f"Invalid side: {side!r}. Expected 'left' or 'right'.")


def _reduce(
    array: ArrayLike,
    func: Callable[[NDArray[Any]], NDArray[Any]],
    /,
    *,
    axis: Sequence[int] | int | None = None,
    keepdims: bool = False,
) -> NDArray[Any] | Any:
    """Apply a core reduction function to the trailing dimension of an array.

    This helper handles the boilerplate of moving the specified axes to the end,
    reshaping the array to a flattened target dimension, applying the specific
    reduction logic, and restoring the original shape if ``keepdims`` is True.

    Args:
        array: Input array to be reduced.
        func: Callable that takes an array with the target axes flattened
            to the last dimension and returns a reduced array.
        axis: Axis or axes along which the reduction is performed.
        keepdims: Whether to retain the reduced axes as dimensions with size one.

    Returns:
        A new array (or scalar) holding the reduced values.
    """
    array = np.asarray(array)
    axes = np.arange(array.ndim)

    if axis is None:
        axes_reduced = axes
    else:
        axes_reduced = np.atleast_1d(axis)

    if array.ndim:
        axes_reduced = np.unique(axes_reduced % array.ndim)
    else:
        axes_reduced = np.unique(axes_reduced)

    axes_remaining = np.setdiff1d(axes, axes_reduced)
    shape_remaining = np.array(array.shape)[axes_remaining]

    reduced = func(
        np.moveaxis(
            array,
            axes_remaining,
            range(len(axes_remaining)),
        ).reshape(*shape_remaining, -1)
    )

    if keepdims:
        shape = np.where(np.isin(axes, axes_reduced), 1, array.shape)
        return reduced.reshape(shape)
    else:
        return reduced if shape_remaining.size else reduced[()]
