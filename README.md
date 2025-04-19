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

### Array comparison

ndtools provides `TotalEquality` and `TotalOrdering` that implement missing equality and ordering operations for multidimensional arrays, respectively.
`TotalEquality` will implement missing `__ne__` from user-defined `__eq__` or missing `__eq__` from user-defined `__ne__`.
The following example implements an equatable object that checks whether each array element is even or not:
```python
import numpy as np
from ndtools import TotalEquality

class Even(TotalEquality):
    def __eq__(self, array):
        return array % 2 == 0

Even() == np.arange(3)  # -> array([True, False, True])
Even() != np.arange(3)  # -> array([False, True, False])
```
It also supports a more intuitive notation with the array written on the left-hand side and the object on the right-hand side:
```python
np.arange(3) == Even()  # -> array([True, False, True])
np.arange(3) != Even()  # -> array([False, True, False])
```

`TotalOrdering` will implement missing ordering operators (`__ge__`, `__gt__`, `__le__`, `__lt__`).
Similar to [`functools.total_ordering`](https://docs.python.org/3/library/functools.html#functools.total_ordering), at least one of them, and `__eq__` or `__ne__` must be user-defined.
The following example implements an equatable object that defines equivalence with a certain range:
```python
import numpy as np
from dataclasses import dataclass
from ndtools import TotalOrdering

@dataclass
class Range(TotalOrdering)
    lower: float
    upper: float

    def __eq__(self, array):
        return (array >= self.lower) & (array < self.upper)

    def __ge__(self, array):
        return array < self.upper

Range(1, 2) == np.arange(3)  # -> array([False, True, False])
Range(1, 2) < np.arange(3)   # -> array([False, False, True])
Range(1, 2) > np.arange(3)   # -> array([True, False, False])
```
It also supports a more intuitive notation with the array written on the left-hand side and the object on the right-hand side:
```python
np.arange(3) == Range(1, 2) # -> array([False, True, False])
np.arange(3) < Range(1, 2)  # -> array([True, False, False])
np.arange(3) > Range(1, 2)  # -> array([False, False, True])
```

### Equatable combination

ndtools provides `All`, `Any`, and `Combinable` that implement logical operations between equatable objects.
Equatable classes that inherit from `Combinable` can perform logical operations between the class instance and other equatable object.
Then ``instance & object`` will return ``All([instance, other])`` and ``instance | object`` will return ``Any[instance, other])``.

```python
import numpy as np
from ndtools import Combinable, TotalEquality

class Even(Combinable, TotalEquality):
    def __eq__(self, array):
        return array % 2 == 0

class Odd(Combinable, TotalEquality):
    def __eq__(self, array):
        return array % 2 == 1

Even() & Odd()  # -> All([Even(), Odd()])
Even() | Odd()  # -> Any([Even(), Odd()])

np.arange(3) == Even() & Odd()  # -> array([False, False, False])
np.arange(3) == Even() | Odd()  # -> array([True, True, True])
```
