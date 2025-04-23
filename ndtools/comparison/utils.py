__all__ = []


# standard library
from typing import Any


def has_usermethod(obj: Any, name: str, /) -> bool:
    """Check if an object has a user-defined method with given name."""
    return (
        hasattr(obj, name)
        and not is_abstractmethod(getattr(obj, name))
        and not is_objectmethod(getattr(obj, name))
    )


def is_abstractmethod(method: Any, /) -> bool:
    """Check if given method is an abstract method."""
    return bool(getattr(method, "__isabstractmethod__", None))


def is_objectmethod(method: Any, /) -> bool:
    """Check if given method is defined in the object class."""
    return method is getattr(object, method.__name__, None)
