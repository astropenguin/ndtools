__all__ = ["get_method", "has_method"]


# standard library
from typing import Any


def get_method(cls: Any, name: str, default: Any, /) -> Any:
    """Return a user-defined method of a class with given name."""
    return getattr(cls, name) if has_method(cls, name) else default


def has_method(cls: Any, name: str, /) -> bool:
    """Check if a class has a user-defined method with given name."""
    return (
        hasattr(cls, name)
        and not is_abstractmethod(getattr(cls, name))
        and not is_objectmethod(getattr(cls, name))
    )


def is_abstractmethod(method: Any, /) -> bool:
    """Check if given method is an abstract method."""
    return bool(getattr(method, "__isabstractmethod__", None))


def is_objectmethod(method: Any, /) -> bool:
    """Check if given method is defined in the object class."""
    return method is getattr(object, method.__name__, None)
