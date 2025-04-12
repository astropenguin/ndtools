# ndtools

[![Release](https://img.shields.io/pypi/v/ndtools?label=Release&color=cornflowerblue&style=flat-square)](https://pypi.org/project/ndtools/)
[![Python](https://img.shields.io/pypi/pyversions/ndtools?label=Python&color=cornflowerblue&style=flat-square)](https://pypi.org/project/ndtools/)
[![Downloads](https://img.shields.io/pypi/dm/ndtools?label=Downloads&color=cornflowerblue&style=flat-square)](https://pepy.tech/project/ndtools)
[![Tests](https://img.shields.io/github/actions/workflow/status/astropenguin/ndtools/tests.yaml?label=Tests&style=flat-square)](https://github.com/astropenguin/ndtools/actions)

Collection of tools to extend multidimensional array operations

## Installation

```shell
pip install ndtools
```

## Usage

### Comparison

ndtools provides `total_equality` and `total_ordering` class decorators that fill in missing multidimensional equality and ordering methods, respectively.
`total_equality` will fill in missing `__ne__` from user-defined `__eq__` or missing `__eq__` from user-defined `__ne__`.
The following example implements an object that checks whether each array element is even or not:
```python
import numpy as np
from ndtools import total_equality

@total_equality
class Even:
    def __eq__(self, array):
        return ~((array % 2).astype(bool))

even = Even()
even == np.arange(3)  # -> array([True, False, True])
even != np.arange(3)  # -> array([False, True, False])
```
It also supports a more intuitive notation with the array written on the left-hand side and the object on the right-hand side:
```python
np.arange(3) == even  # -> array([True, False, True])
np.arange(3) != even  # -> array([False, True, False])
```

`total_ordering` will fill in missing ordering operators (`__ge__`, `__gt__`, `__le__`, `__lt__`).
As with [`functools.total_ordering`](https://docs.python.org/3/library/functools.html#functools.total_ordering), at least one of them, and `__eq__` or `__ne__` must be user-defined.
The following example implements an interval object that defines equivalence with a certain range:
```python
import numpy as np
from dataclasses import dataclass
from ndtools import total_ordering

@dataclass
@total_ordering
class Interval:
    lower: float
    upper: float

    def __eq__(self, array):
        return (array >= self.lower) & (array < self.upper)

    def __ge__(self, array):
        return array < self.upper

Interval(1, 2) == np.arange(3)  # -> array([False, True, False])
Interval(1, 2) < np.arange(3)   # -> array([False, False, True])
Interval(1, 2) > np.arange(3)   # -> array([True, False, False])
```
It also supports a more intuitive notation with the array written on the left-hand side and the object on the right-hand side:
```python
np.arange(3) == Interval(1, 2) # -> array([False, True, False])
np.arange(3) < Interval(1, 2)  # -> array([True, False, False])
np.arange(3) > Interval(1, 2)  # -> array([False, False, True])
```
