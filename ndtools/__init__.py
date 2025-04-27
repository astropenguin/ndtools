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
)
from .comparison import (
    All,
    Any,
    Apply,
    Combinable,
    Equatable,
    Not,
    Match,
    Range,
    Orderable,
)
