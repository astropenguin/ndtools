__all__ = [
    "ANY",
    "NEVER",
    "All",
    "Any",
    "Apply",
    "Combinable",
    "Equatable",
    "Match",
    "Not",
    "Range",
    "Orderable",
    "comparison",
]
__version__ = "0.3.0"


# dependencies
from . import comparison
from .comparison.builtins import (
    ANY,
    NEVER,
    Apply,
    Match,
    Range,
)
from .comparison.comparables import (
    All,
    Any,
    Combinable,
    Equatable,
    Not,
    Orderable,
)
