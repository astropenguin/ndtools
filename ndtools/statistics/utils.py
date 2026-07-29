__all__ = [
    "DuckArray",
    "asduckarray",
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
from typing import Any, Literal, Protocol, cast, runtime_checkable

# dependencies
import numpy as np
from numpy.exceptions import AxisError
from numpy.typing import ArrayLike, NDArray

# constants
FIRST_AXIS = 0
FIRST_INDEX = 0
LAST_AXIS = -1
LAST_INDEX = -1


@runtime_checkable
class DuckArray(Protocol):
    """Protocol for objects exposing basic multidimensional array attributes.

    This protocol defines the minimal set of attributes required for an object
    to be safely treated as a multidimensional array in numerical computations.
    """

    @property
    def dtype(self) -> Any: ...
    @property
    def ndim(self) -> int: ...
    @property
    def shape(self) -> tuple[int, ...]: ...


def asduckarray(array: ArrayLike | DuckArray, /) -> NDArray[Any]:
    """Return the input as an array compatible with DuckArray.

    This function avoids triggering unnecessary evaluation of lazy arrays
    by checking for DuckArray compatibility before falling back to NumPy.
    The result is type-cast to a ``numpy.ndarray`` to ensure compatibility
    with downstream NumPy type hints.

    Args:
        array: Input array or object that can be converted to an array.

    Returns:
        The original object if it satisfies the DuckArray requirements,
        or a new NumPy array otherwise.
    """
    if isinstance(array, DuckArray):
        return cast(NDArray[Any], array)
    else:
        return np.asarray(array)


def first(
    array: ArrayLike | DuckArray,
    /,
    axis: Sequence[int] | int | None = None,
    keepdims: bool = False,
) -> Any:
    """Compute the first element along the specified axis.

    Args:
        array: Input array or object that can be converted to an array.
        axis: Axis or axes along which the first element is computed.
        keepdims: Whether to retain the reduced axes as dimensions with size one.

    Returns:
        A new array (or scalar) holding the computed first element.
    """

    def func(agg: NDArray[Any], /) -> NDArray[Any]:
        indices = np.full(agg.shape[:LAST_AXIS], FIRST_INDEX)[..., np.newaxis]
        return np.take_along_axis(agg, indices, LAST_AXIS).squeeze(LAST_AXIS)

    return _reduce(asduckarray(array), func, axis=axis, keepdims=keepdims)


def nanfirst(
    array: ArrayLike | DuckArray,
    /,
    axis: Sequence[int] | int | None = None,
    keepdims: bool = False,
) -> Any:
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

    def func(agg: NDArray[Any], /) -> NDArray[Any]:
        indices = np.argmax(~np.isnan(agg), axis=LAST_AXIS)[..., np.newaxis]
        return np.take_along_axis(agg, indices, LAST_AXIS).squeeze(LAST_AXIS)

    return _reduce(asduckarray(array), func, axis=axis, keepdims=keepdims)


def last(
    array: ArrayLike | DuckArray,
    /,
    axis: Sequence[int] | int | None = None,
    keepdims: bool = False,
) -> Any:
    """Compute the last element along the specified axis.

    Args:
        array: Input array or object that can be converted to an array.
        axis: Axis or axes along which the last element is computed.
        keepdims: Whether to retain the reduced axes as dimensions with size one.

    Returns:
        A new array (or scalar) holding the computed last element.
    """

    def func(agg: NDArray[Any], /) -> NDArray[Any]:
        indices = np.full(agg.shape[:LAST_AXIS], LAST_INDEX)[..., np.newaxis]
        return np.take_along_axis(agg, indices, LAST_AXIS).squeeze(LAST_AXIS)

    return _reduce(asduckarray(array), func, axis=axis, keepdims=keepdims)


def nanlast(
    array: ArrayLike | DuckArray,
    /,
    axis: Sequence[int] | int | None = None,
    keepdims: bool = False,
) -> Any:
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

    def func(agg: NDArray[Any], /) -> NDArray[Any]:
        indices = (
            # fmt: off
            - np.argmax(~np.isnan(agg)[..., ::-1], LAST_AXIS)[..., np.newaxis]
            + (agg.shape[LAST_AXIS] - 1)
            # fmt: on
        )
        return np.take_along_axis(agg, indices, LAST_AXIS).squeeze(LAST_AXIS)

    return _reduce(asduckarray(array), func, axis=axis, keepdims=keepdims)


def mad(
    array: ArrayLike | DuckArray,
    /,
    axis: Sequence[int] | int | None = None,
    keepdims: bool = False,
) -> Any:
    """Compute the median absolute deviation (MAD) along the specified axis.

    Args:
        array: Input array or object that can be converted to an array.
        axis: Axis or axes along which the MAD is computed.
        keepdims: Whether to retain the reduced axes as dimensions with size one.

    Returns:
        A new array (or scalar) holding the computed MAD.
    """
    array = asduckarray(array)
    median = np.median

    return median(
        np.abs(array - median(array, axis=axis, keepdims=True)),
        axis=axis,
        keepdims=keepdims,
    )


def nanmad(
    array: ArrayLike | DuckArray,
    /,
    axis: Sequence[int] | int | None = None,
    keepdims: bool = False,
) -> Any:
    """Compute the median absolute deviation (MAD) along the specified axis, ignoring NaNs.

    Args:
        array: Input array or object that can be converted to an array.
        axis: Axis or axes along which the MAD (ignoreing NaNs) is computed.
        keepdims: Whether to retain the reduced axes as dimensions with size one.

    Returns:
        A new array (or scalar) holding the computed MAD.
    """
    array = asduckarray(array)
    median = np.nanmedian

    return median(
        np.abs(array - median(array, axis=axis, keepdims=True)),
        axis=axis,
        keepdims=keepdims,
    )


def middle(
    array: ArrayLike | DuckArray,
    /,
    axis: Sequence[int] | int | None = None,
    keepdims: bool = False,
) -> Any:
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

    def func(agg: NDArray[Any], /) -> NDArray[Any]:
        middle_index = (agg.shape[LAST_AXIS] - 1) // 2
        indices = np.full(agg.shape[:LAST_AXIS], middle_index)[..., np.newaxis]
        return np.take_along_axis(agg, indices, LAST_AXIS).squeeze(LAST_AXIS)

    return _reduce(asduckarray(array), func, axis=axis, keepdims=keepdims)


def nanmiddle(
    array: ArrayLike | DuckArray,
    /,
    axis: Sequence[int] | int | None = None,
    side: Literal["left", "right"] = "left",
    keepdims: bool = False,
) -> Any:
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

    def func_left(agg: NDArray[Any], /) -> NDArray[Any]:
        agg = agg[..., : (agg.shape[LAST_AXIS] - 1) // 2 + 1]
        indices = (
            # fmt: off
            - np.argmax(~np.isnan(agg)[..., ::-1], LAST_AXIS)[..., np.newaxis]
            + (agg.shape[LAST_AXIS] - 1)
            # fmt: on
        )
        return np.take_along_axis(agg, indices, LAST_AXIS).squeeze(LAST_AXIS)

    def func_right(agg: NDArray[Any], /) -> NDArray[Any]:
        agg = agg[..., (agg.shape[LAST_AXIS] - 1) // 2 :]
        indices = np.argmax(~np.isnan(agg), axis=LAST_AXIS)[..., np.newaxis]
        return np.take_along_axis(agg, indices, LAST_AXIS).squeeze(LAST_AXIS)

    if side == "left":
        return _reduce(asduckarray(array), func_left, axis=axis, keepdims=keepdims)
    elif side == "right":
        return _reduce(asduckarray(array), func_right, axis=axis, keepdims=keepdims)
    else:
        raise ValueError(f"Invalid side: {side!r}. Expected 'left' or 'right'.")


def _reduce(
    array: NDArray[Any],
    func: Callable[[NDArray[Any]], NDArray[Any]],
    /,
    *,
    axis: Sequence[int] | int | None,
    keepdims: bool,
) -> Any:
    """Apply a core reduction function to the trailing dimension of an array.

    This helper handles the boilerplate of moving the specified axes to the end,
    reshaping the array to a flattened target dimension, applying the specific
    reduction logic, and restoring the original shape if ``keepdims`` is True.

    Args:
        array: Input array to be reduced.
        func: Callable that takes an array with the target axes flattened
            to the last dimension, and returns a reduced array.
        axis: Axis or axes along which the reduction is performed.
        keepdims: Whether to retain the reduced axes as dimensions with size one.

    Returns:
        A new array (or scalar) holding the reduced values.
    """
    info = _reduce_info(array, axis=axis)
    reduced = func(
        np.moveaxis(
            array,
            info["axes_remaining"],
            range(len(info["axes_remaining"])),
        ).reshape(*info["shape_remaining"], -1)
    )

    if keepdims:
        return reduced.reshape(info["shape"])

    if info["shape_remaining"].size:
        return reduced

    return reduced[()]


def _reduce_info(
    array: NDArray[Any],
    /,
    *,
    axis: Sequence[int] | int | None = None,
) -> dict[str, NDArray[np.int64]]:
    """Calculate axis and shape information for generalized array reduction.

    Args:
        array: Input array to extract information from.
        axis: Target axis or axes for the reduction.

    Returns:
        A dictionary containing the following keys with axis and shape information.

        - ``'axes'``: Axes of the reduced array,
          assuming the reduced axes are retained.
        - ``'axes_reduced'``: Reduced axes of the input array after reduction.
        - ``'axes_remaining'``: Remaining axes of the input array after reduction.
        - ``'shape'``: Shape of the reduced array,
          assuming the reduced axes are retained as dimensions with size one.
        - ``'shape_reduced'``: Shape of the input array along the reduced axes.
        - ``'shape_remaining'``: Shape of the input array along the remaining axes.

    """
    shape = np.array(array.shape, np.int64)
    axes = np.arange(array.ndim, dtype=np.int64)

    if axis is None:
        axes_reduced = axes
    else:
        axes_reduced = np.atleast_1d(axis).astype(np.int64)

    if array.ndim:
        if (axes_reduced < -array.ndim).any():
            raise AxisError(
                f"Axis {axis!r} is out of bounds for array of dimension {array.ndim!r}"
            )

        if (axes_reduced >= array.ndim).any():
            raise AxisError(
                f"Axis {axis!r} is out of bounds for array of dimension {array.ndim!r}"
            )

        axes_reduced = np.unique(axes_reduced % array.ndim)
    else:
        axes_reduced = np.unique(axes_reduced)

    return {
        "axes": axes,
        "axes_reduced": axes_reduced,
        "axes_remaining": (axes_remaining := np.setdiff1d(axes, axes_reduced)),
        "shape": np.where(np.isin(axes, axes_reduced), 1, shape),
        "shape_reduced": shape[axes_reduced],
        "shape_remaining": shape[axes_remaining],
    }
