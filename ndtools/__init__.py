__all__ = [
    "ANY",
    "NEVER",
    "All",
    "Any",
    "Combinable",
    "Equatable",
    "Match",
    "Not",
    "Range",
    "Orderable",
    "Where",
    "comparison",
]
__version__ = "1.0.0"


# dependencies
from . import comparison
from .comparison.builtins import (
    ANY,
    NEVER,
    Match,
    Range,
    Where,
)
from .comparison.comparables import (
    All,
    Any,
    Combinable,
    Equatable,
    Not,
    Orderable,
)
