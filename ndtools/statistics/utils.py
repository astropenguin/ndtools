__all__ = ["DuckArray", "first", "nanfirst", "last", "nanlast", "mad", "nanmad"]

# standard library
from collections.abc import Callable, Sequence
from typing import Any, Protocol, cast, runtime_checkable

# dependencies
import numpy as np
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


def first(
    a: ArrayLike | DuckArray,
    /,
    axis: Sequence[int] | int | None = None,
    keepdims: bool = False,
) -> NDArray[Any] | Any:
    """Compute the first element along the specified axis.

    Args:
        a: Input array or object that can be converted to an array.
        axis: Axis or axes along which the first element is computed.
        keepdims: Whether to retain the reduced axes as dimensions with size one.

    Returns:
        A new array (or scalar) holding the computed first element.
    """

    def func(agg: NDArray[Any], /) -> NDArray[Any]:
        indices = np.full(agg.shape[:LAST_AXIS], FIRST_INDEX)[..., np.newaxis]
        return np.take_along_axis(agg, indices, LAST_AXIS).squeeze(LAST_AXIS)

    return _apply(_asndarray(a), func, axis=axis, keepdims=keepdims)


def nanfirst(
    a: ArrayLike | DuckArray,
    /,
    axis: Sequence[int] | int | None = None,
    keepdims: bool = False,
) -> NDArray[Any] | Any:
    """Compute the first element along the specified axis, ignoring NaNs.

    Args:
        a: Input array or object that can be converted to an array.
        axis: Axis or axes along which the first non-NaN element is computed.
        keepdims: Whether to retain the reduced axes as dimensions with size one.

    Returns:
        A new array (or scalar) holding the computed first element.
    """

    def func(agg: NDArray[Any], /) -> NDArray[Any]:
        indices = np.argmax(~np.isnan(agg), axis=LAST_AXIS)[..., np.newaxis]
        return np.take_along_axis(agg, indices, LAST_AXIS).squeeze(LAST_AXIS)

    return _apply(_asndarray(a), func, axis=axis, keepdims=keepdims)


def last(
    a: ArrayLike | DuckArray,
    /,
    axis: Sequence[int] | int | None = None,
    keepdims: bool = False,
) -> NDArray[Any] | Any:
    """Compute the last element along the specified axis.

    Args:
        a: Input array or object that can be converted to an array.
        axis: Axis or axes along which the last element is computed.
        keepdims: Whether to retain the reduced axes as dimensions with size one.

    Returns:
        A new array (or scalar) holding the computed last element.
    """

    def func(agg: NDArray[Any], /) -> NDArray[Any]:
        indices = np.full(agg.shape[:LAST_AXIS], LAST_INDEX)[..., np.newaxis]
        return np.take_along_axis(agg, indices, LAST_AXIS).squeeze(LAST_AXIS)

    return _apply(_asndarray(a), func, axis=axis, keepdims=keepdims)


def nanlast(
    a: ArrayLike | DuckArray,
    /,
    axis: Sequence[int] | int | None = None,
    keepdims: bool = False,
) -> NDArray[Any] | Any:
    """Compute the last element along the specified axis, ignoring NaNs.

    Args:
        a: Input array or object that can be converted to an array.
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

    return _apply(_asndarray(a), func, axis=axis, keepdims=keepdims)


def mad(
    a: ArrayLike | DuckArray,
    /,
    axis: Sequence[int] | int | None = None,
    keepdims: bool = False,
) -> Any:
    """Compute the median absolute deviation (MAD) along the specified axis.

    Args:
        a: Input array or object that can be converted to an array.
        axis: Axis or axes along which the MAD is computed.
        keepdims: Whether to retain the reduced axes as dimensions with size one.

    Returns:
        A new array (or scalar) holding the computed MAD.
    """
    array = _asndarray(a)
    median = np.median

    return median(
        np.abs(array - median(array, axis=axis, keepdims=True)),
        axis=axis,
        keepdims=keepdims,
    )


def nanmad(
    a: ArrayLike | DuckArray,
    /,
    axis: Sequence[int] | int | None = None,
    keepdims: bool = False,
) -> Any:
    """Compute the median absolute deviation (MAD) along the specified axis, ignoring NaNs.

    Args:
        a: Input array or object that can be converted to an array.
        axis: Axis or axes along which the MAD (ignoreing NaNs) is computed.
        keepdims: Whether to retain the reduced axes as dimensions with size one.

    Returns:
        A new array (or scalar) holding the computed MAD.
    """
    array = _asndarray(a)
    median = np.nanmedian

    return median(
        np.abs(array - median(array, axis=axis, keepdims=True)),
        axis=axis,
        keepdims=keepdims,
    )


def _asndarray(a: ArrayLike | DuckArray, /) -> NDArray[Any]:
    """Safely return the input as a NumPy array or compatible DuckArray.

    This function avoids triggering unnecessary evaluation of lazy arrays
    by checking for DuckArray compatibility before falling back to NumPy.
    The result is type-cast to a ``numpy.ndarray`` to ensure compatibility
    with downstream NumPy type hints.

    Args:
        a: Input array or object that can be converted to an array.

    Returns:
        The original object if it satisfies the DuckArray requirements,
        or a new NumPy array otherwise.
    """
    if isinstance(a, DuckArray):
        return cast(NDArray[Any], a)
    else:
        return np.asarray(a)


def _apply(
    array: NDArray[Any],
    func: Callable[[NDArray[Any]], NDArray[Any]],
    /,
    *,
    axis: Sequence[int] | int | None,
    keepdims: bool,
) -> NDArray[Any] | Any:
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
    meta = _meta(array, axis=axis)
    reduced = func(
        np.moveaxis(
            array,
            meta["axes_remaining"],
            range(len(meta["axes_remaining"])),
        ).reshape(*meta["shape_remaining"], -1)
    )

    if keepdims:
        return reduced.reshape(meta["shape"])

    if meta["shape_remaining"].size:
        return reduced

    return reduced[()]


def _meta(
    array: NDArray[Any],
    /,
    *,
    axis: Sequence[int] | int | None = None,
) -> dict[str, NDArray[np.int64]]:
    """Gather axis and shape metadata required for generalized array reduction.

    Args:
        array: Input array to extract metadata from.
        axis: Target axis or axes for the reduction.

    Returns:
        A dictionary containing the following keys with axis and shape metadata.

        * ``'axes'``: Axes of the reduced array,
          assuming the reduced axes are retained.
        * ``'axes_reduced'``: Reduced axes of the input array after reduction.
        * ``'axes_remaining'``: Remaining axes of the input array after reduction.
        * ``'shape'``: Shape of the reduced array,
          assuming the reduced axes are retained as dimensions with size one.
        * ``'shape_reduced'``: Shape of the input array along the reduced axes.
        * ``'shape_remaining'``: Shape of the input array along the remaining axes.

    """
    shape = np.array(array.shape, np.int64)
    axes = np.arange(array.ndim, dtype=np.int64)

    if axis is None:
        axes_reduced = axes
    else:
        axes_reduced = np.atleast_1d(axis).astype(np.int64)

    if array.ndim:
        axes_reduced = axes_reduced % array.ndim

    axes_remaining = np.setdiff1d(axes, axes_reduced)

    return {
        "axes": axes,
        "axes_reduced": axes_reduced,
        "axes_remaining": axes_remaining,
        "shape": np.where(np.isin(axes, axes_reduced), 1, shape),
        "shape_reduced": shape[axes_reduced],
        "shape_remaining": shape[axes_remaining],
    }
