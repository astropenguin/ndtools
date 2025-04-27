# ndtools

[![Release](https://img.shields.io/pypi/v/ndtools?label=Release&color=cornflowerblue&style=flat-square)](https://pypi.org/project/ndtools/)
[![Python](https://img.shields.io/pypi/pyversions/ndtools?label=Python&color=cornflowerblue&style=flat-square)](https://pypi.org/project/ndtools/)
[![Downloads](https://img.shields.io/pypi/dm/ndtools?label=Downloads&color=cornflowerblue&style=flat-square)](https://pepy.tech/project/ndtools)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.15291176-cornflowerblue?style=flat-square)](https://doi.org/10.5281/zenodo.15291176)
[![Tests](https://img.shields.io/github/actions/workflow/status/astropenguin/ndtools/tests.yaml?label=Tests&style=flat-square)](https://github.com/astropenguin/ndtools/actions)

Collection of tools to extend multidimensional array operations

## Installation

```shell
pip install ndtools
```

## Usage

ndtools allows you to compare NumPy arrays, pandas Series, xarray DataArrays, and other array-like objects (often called "duck arrays") against custom conditions in an intuitive way.
It achieves this broad compatibility by leveraging standard protocols like `__array_ufunc__`, ensuring that the comparison logic works seamlessly across different libraries as long as they adhere to these conventions.

### Core concepts

At its core, ndtools uses mixin classes to make your own objects comparable with these duck arrays.
This allows you to define complex, domain-specific comparison logic that goes beyond simple value checks, while remaining compatible with the wider Python data science ecosystem.

#### `Equatable` mixin

Implement `Equatable` mixin when you need custom equality logic (`==`, `!=`).
Simply define `__eq__` (or `__ne__`) on your class, specifying how it should compare against an array's elements.
ndtools leverages NumPy's `__array_ufunc__` protocol behind the scenes, ensuring that comparisons like `array == YourClass()` and `YourClass() == array` both work seamlessly and symmetrically across compatible array types.
Crucially, ndtools also automatically derives the missing comparison operator for you (e.g., it creates a working `__ne__` if you only provide `__eq__`), reducing boilerplate code.

```python
import numpy as np
from ndtools import Equatable

class Even(Equatable):
    def __eq__(self, array):
        return array % 2 == 0

Even() == np.arange(3)  # -> array([True, False, True])
np.arange(3) == Even()  # -> array([True, False, True])

Even() != np.arange(3)  # -> array([False, True, False])
np.arange(3) != Even()  # -> array([False, True, False])
```

#### `Orderable` mixin

For comparisons involving order (`>=`, `>`, `<=`, `<`), inherit from `Orderable` mixin.
Similar in spirit to Python's standard library `functools.total_ordering`, `Orderable` significantly simplifies defining ordered comparisons.
You only need to implement one ordering method (e.g., `__gt__`) and one equality method (`__eq__` or `__ne__`).
From this minimal definition, ndtools automatically and robustly derives all other comparison operators (`<`, `<=`, `>=`, `!=` if needed) based on logical equivalences (e.g., `a <= b` is equivalent to `not (a > b)`), again using `__array_ufunc__` for broad compatibility.
This powerful mechanism allows you to implement custom sorting criteria or range-like checks with minimal code, while ensuring consistent behavior across all six comparison operators.
Like `Equatable`, it ensures comparisons work symmetrically (e.g., both `array > YourClass()` and `YourClass() < array` work correctly).

```python
import numpy as np
from dataclasses import dataclass
from ndtools import Orderable

@dataclass
class Range(Orderable):
    lower: float
    upper: float

    def __eq__(self, array):
        return (array >= self.lower) & (array < self.upper)

    def __ge__(self, array):
        return array < self.upper

Range(1, 2) == np.arange(3)  # -> array([False, True, False])
np.arange(3) == Range(1, 2)  # -> array([False, True, False])

Range(1, 2) >= np.arange(3)  # -> array([True, True, False])
np.arange(3) <= Range(1, 2)  # -> array([True, True, False])
```

#### Combining comparables

Multiple comparables can be combined using standard Python logical operators.
This applies to any comparable object inheriting from `Combinable`, including built-in comparables and primitive types.
Use `&` (AND) for logical conjunction (all conditions must be `True`) and `|` (OR) for logical disjunction (at least one condition must be `True`).
You can also use `Not(comparable)` to invert the result of another comparable object.
This allows for the construction of complex, readable query expressions.
The comparison is evaluated element-wise when the combined condition is compared against an array.
Note that primitive types like `0` in the examples below are implicitly treated as comparable values when combined using `&` or `|`.

```python
import numpy as np
from ndtools import Combinable, Equatable

class Even(Combinable, Equatable):
    def __eq__(self, array):
        return array % 2 == 0

    def __ne__(self, array):
        return ~(self == array)

class Odd(Combinable, Equatable):
    def __eq__(self, array):
        return array % 2 == 1

    def __ne__(self, array):
        return ~(self == array)

Even() | Odd()  # -> Any([Even(), Odd()])
Even() & Odd()  # -> All([Even(), Odd()])
np.arange(3) == Even() | Odd()  # -> array([True, True, True])
np.arange(3) == Even() & Odd()  # -> array([False, False, False])
np.arange(3) == Not(1)  # -> array([True, False, True])
```

### Built-in comparables

ndtools provides several ready-to-use comparable objects designed for duck arrays.

#### `ANY` / `NEVER`

Comparison with them always evaluates to `True` or `False`, respectively.

```python
import numpy as np
from ndtools import ANY, NEVER

np.arange(3) == ANY  # -> array([True, True, True])
np.arange(3) == NEVER  # -> array([False, False, False])
```

#### `Match(pat, case=True, flags=0, na=None)`

Checks if string array elements fully match a regular expression pattern (uses `pandas.Series.str.fullmatch`).

```python
import numpy as np
from ndtools import Match

np.array(["a", "aa"]) == Match("a+")  # -> array([True, True])
```

#### `Range(lower, upper, bounds="[)")`

Checks if array elements are within a specified range.
`bounds` controls inclusivity (`[)`, `[]`, `(]`, `()`).
Use `None` for unbounded sides.

```python
import numpy as np
from ndtools import Range

np.arange(3) == Range(1, 2) # -> array([False, True, False])
np.arange(3) < Range(1, 2)  # -> array([True, False, False])
np.arange(3) > Range(1, 2)  # -> array([False, False, True])

np.arange(3) == Range(None, 2)  # -> array([True, True, False])
np.arange(3) == Range(1, None)  # -> array([False, True, True])
np.arange(3) == Range(None, None)  # -> array([True, True, True])
```

#### `Where(func, *args, **kwargs)`

Checks if `func(array, *args, **kwargs)` returns `True` for array elements.

```python
import numpy as np
from ndtools import Where
from numpy.char import isupper

np.array(["A", "b"]) == Where(isupper)  # -> array([True, False])
```
