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
    "collections",
    "comparable",
    "comparison",
    "operators",
]
__version__ = "0.3.0"


# dependencies
from . import collections
from . import comparable
from . import comparison
from . import operators
from .collections import *
from .comparable import *
from .comparison import *
