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

### Array comparisons

ndtools provides `Equatable` and `Orderable` that implement equality and ordering operations for multidimensional arrays, respectively.
`Equatable` will implement missing `__ne__` from user-defined `__eq__` or missing `__eq__` from user-defined `__ne__`.
The following example implements a comparable that checks whether each array element is even or not:
```python
import numpy as np
from ndtools import Equatable

class Even(Equatable):
    def __eq__(self, array):
        return array % 2 == 0

np.arange(3) == Even()  # -> array([True, False, True])
np.arange(3) != Even()  # -> array([False, True, False])
```

`Orderable` will implement missing ordering operators (`__ge__`, `__gt__`, `__le__`, `__lt__`).
Similar to [`functools.total_ordering`](https://docs.python.org/3/library/functools.html#functools.total_ordering), at least one of them, and `__eq__` or `__ne__` must be user-defined.
The following example implements a comparable that defines equivalence with a certain range:
```python
import numpy as np
from dataclasses import dataclass
from ndtools import Orderable

@dataclass
class Range(Orderable)
    lower: float
    upper: float

    def __eq__(self, array):
        return (array >= self.lower) & (array < self.upper)

    def __ge__(self, array):
        return array < self.upper

np.arange(3) == Range(1, 2) # -> array([False, True, False])
np.arange(3) < Range(1, 2)  # -> array([True, False, False])
np.arange(3) > Range(1, 2)  # -> array([False, False, True])
```

### Combination of comparables

ndtools provides `All`, `Any`, and `Combinable` that implement logical operations between comparables.
Comparable classes that inherit from `Combinable` can perform logical operations between comparables.
Then ``comparable_0 & comparable_1 & ...`` will return ``All([comparable_0, comparable_1, ...])`` and ``comparable_0 | comparable_1 | ...`` will return ``Any[comparable_0, comparable_1, ...])``.

```python
import numpy as np
from ndtools import Combinable, Equatable

class Even(Combinable, Equatable):
    def __eq__(self, array):
        return array % 2 == 0

class Odd(Combinable, Equatable):
    def __eq__(self, array):
        return array % 2 == 1

Even() & Odd()  # -> All([Even(), Odd()])
Even() | Odd()  # -> Any([Even(), Odd()])

np.arange(3) == Even() & Odd()  # -> array([False, False, False])
np.arange(3) == Even() | Odd()  # -> array([True, True, True])
```
