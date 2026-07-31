__all__ = ["DuckArray", "asduckarray"]

# standard library
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
