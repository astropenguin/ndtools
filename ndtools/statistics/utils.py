__all__ = ["DuckArray", "mad", "nanmad"]

# standard library
from collections.abc import Sequence
from typing import Any, Protocol, cast, runtime_checkable

# dependencies
import numpy as np
from numpy.typing import ArrayLike, NDArray


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
            The default is to compute the MAD of the flattened array.
        keepdims: If this is set to ``True``, the axes which are reduced
            are left in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the original array.

    Returns:
        A new array (or scalar) holding the computed MAD.
    """
    if not isinstance(a, DuckArray):
        array = np.asarray(a)
    else:
        array = cast(NDArray[Any], a)

    return np.median(
        np.abs(array - np.median(array, axis=axis, keepdims=True)),
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
        axis: Axis or axes along which the MAD is computed.
            The default is to compute the MAD of the flattened array.
        keepdims: If this is set to ``True``, the axes which are reduced
            are left in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the original array.

    Returns:
        A new array (or scalar) holding the computed MAD.
    """
    if not isinstance(a, DuckArray):
        array = np.asarray(a)
    else:
        array = cast(NDArray[Any], a)

    return np.nanmedian(
        np.abs(array - np.nanmedian(array, axis=axis, keepdims=True)),
        axis=axis,
        keepdims=keepdims,
    )
