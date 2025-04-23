__all__ = [
    "All",
    "Any",
    "Apply",
    "Combinable",
    "Equatable",
    "Match",
    "Range",
    "Orderable",
    "TotalEquality",
    "TotalOrdering",
    "comparison",
]
__version__ = "0.3.0"


# dependencies
from . import comparison
from .comparison.collections import *
from .comparison.comparable import *
