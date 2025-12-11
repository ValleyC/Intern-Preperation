# Deep-ML Problem Notes

## Table of Contents

### Concepts & Reference
- [Key Concepts](#key-concepts)
  - [Python Type Hints](#python-type-hints)
  - [The `pass` Keyword](#the-pass-keyword)
  - [`sum()` Function](#sum-function)
  - [`zip()` and `*` Unpacking](#zip-and--unpacking)
  - [Shape as Tuple](#shape-as-tuple)
  - [Python `elif` (not `else if`)](#python-elif-not-else-if)
- [Framework Comparison](#framework-comparison)
  - [When to Use What](#when-to-use-what)
  - [Common Operations](#common-operations)
- [Quick Reference](#quick-reference)

### Problems Solved (11 Total)

| # | Problem | Difficulty | Category |
|---|---------|------------|----------|
| 1 | [Matrix-Vector Dot Product](#1-matrix-vector-dot-product-easy---linear-algebra) | Easy | Linear Algebra |
| 2 | [Transpose Matrix](#2-transpose-matrix-easy---linear-algebra) | Easy | Linear Algebra |
| 3 | [Reshape Matrix](#3-reshape-matrix-easy---linear-algebra) | Easy | Linear Algebra |
| 4 | [Calculate Mean by Row or Column](#4-calculate-mean-by-row-or-column-easy---linear-algebra) | Easy | Linear Algebra |
| 5 | [Scalar Multiplication of a Matrix](#5-scalar-multiplication-of-a-matrix-easy---linear-algebra) | Easy | Linear Algebra |
| 6 | [2x2 Matrix Eigenvalues](#6-2x2-matrix-eigenvalues-easy---linear-algebra) | Easy | Linear Algebra |
| 7 | [Matrix Transformation T⁻¹AS](#7-matrix-transformation-t⁻¹as-easy---linear-algebra) | Easy | Linear Algebra |
| 8 | [2x2 Matrix Inverse](#8-2x2-matrix-inverse-medium---linear-algebra) | Medium | Linear Algebra |
| 9 | [Matrix Multiplication](#9-matrix-multiplication-medium---linear-algebra) | Medium | Linear Algebra |
| 10 | [Covariance Matrix](#10-covariance-matrix-medium---linear-algebra) | Medium | Linear Algebra |
| 11 | [Jacobi Method](#11-jacobi-method-medium---linear-algebra) | Medium | Linear Algebra |

---

## Key Concepts

### Python Type Hints
```python
def func(a: list[int], b: str) -> bool:
#        ^parameter types       ^return type
```
- `->` indicates return type
- `int|float` means "int OR float" (Python 3.10+)
- Not enforced at runtime, but helps with documentation and IDE support
- **Very common in AI/ML production code**

### The `pass` Keyword
- Placeholder that does nothing
- Used when syntax requires a statement but no code needed yet
- Replace with your implementation

### `sum()` Function
```python
sum(5)                        # ❌ TypeError - not iterable
sum([1, 2, 3])                # ✅ Returns 6
sum(x for x in range(3))      # ✅ Generator expression
```
**Common mistake**: `sum(a * b)` when you meant `sum(a[i] * b[i] for i in range(n))`

### `zip()` and `*` Unpacking
```python
# zip() pairs elements from multiple iterables
list(zip([1,2], [3,4]))       # [(1,3), (2,4)]

# * unpacks a list into separate arguments
a = [[1,2,3], [4,5,6]]
zip(*a)                        # Same as zip([1,2,3], [4,5,6])
list(zip(*a))                  # [(1,4), (2,5), (3,6)] - transpose!

# Common pattern for transpose:
[list(row) for row in zip(*matrix)]
```

### Shape as Tuple
```python
# NumPy and PyTorch expect shape as a TUPLE, not separate args
np.zeros(3, 4)          # ❌ Wrong - interpreted as dtype argument
np.zeros((3, 4))        # ✅ Correct - 3x4 matrix

torch.zeros(3, 4)       # ✅ PyTorch accepts both
torch.zeros((3, 4))     # ✅ Also works

# Reshape also expects tuple
arr.reshape(4, 2)       # ✅ Works
arr.reshape((4, 2))     # ✅ Also works
arr.reshape(new_shape)  # ✅ Pass tuple variable directly
```

### Python `elif` (not `else if`)
```python
# Python uses "elif", not "else if"
if mode == 'a':
    do_a()
elif mode == 'b':    # ✅ Correct
    do_b()
else:
    do_default()

# Common mistake from other languages:
else if mode == 'b':  # ❌ SyntaxError in Python
```

| Language | Syntax |
|----------|--------|
| Python | `elif` |
| JavaScript/C/Java | `else if` |
| Ruby | `elsif` |

---

## Framework Comparison

| Feature | NumPy | PyTorch | TensorFlow |
|---------|-------|---------|------------|
| Purpose | Numerical computing | Deep learning | Deep learning |
| GPU Support | No | Yes | Yes |
| Auto-differentiation | No | Yes | Yes |
| Industry Use | Everywhere | Research & Production | Production & Mobile |

### When to Use What
- **NumPy**: Data preprocessing, basic math, non-ML code
- **PyTorch**: Research, prototyping, training models
- **TensorFlow**: Production deployment, mobile apps

### Common Operations
```python
# NumPy
import numpy as np
np.dot(matrix, vector)

# PyTorch
import torch
torch.matmul(matrix, vector)
matrix @ vector  # Same thing, cleaner syntax
```

---

## Problems Solved

### 1. Matrix-Vector Dot Product (Easy - Linear Algebra)

**Problem**: Compute dot product of matrix and vector, return -1 if dimensions incompatible.

**Key Points**:
- Matrix (n×m) requires vector of length m
- Each result element = dot product of one row with the vector

**Pure Python Solution**:
```python
def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
    if not a or not a[0]:
        return -1
    if len(a[0]) != len(b):
        return -1

    result = []
    for row in a:
        dot_product = sum(row[i] * b[i] for i in range(len(b)))
        result.append(dot_product)
    return result
```

**PyTorch Solution**:
```python
import torch

def matrix_dot_vector(a, b) -> torch.Tensor:
    a_t = torch.as_tensor(a, dtype=torch.float)
    b_t = torch.as_tensor(b, dtype=torch.float)

    if a_t.size(1) != b_t.size(0):
        return torch.tensor(-1)

    return torch.matmul(a_t, b_t)
```

**Mistakes Made**:
- ❌ `sum(row[i] * b[i])` - sum() needs an iterable, not a single number
- ✅ `sum(row[i] * b[i] for i in range(len(b)))` - generator expression

**Learned**:
- Type hints syntax (`->`, `list[int|float]`)
- `pass` keyword as placeholder
- Difference between manual loops vs PyTorch's `@` operator

---

### 2. Transpose Matrix (Easy - Linear Algebra)

**Problem**: Compute the transpose of a matrix (flip rows and columns).

**Key Points**:
- Element at `[i][j]` moves to `[j][i]`
- Matrix (m×n) becomes (n×m)

**Pure Python Solution (using zip)**:
```python
def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    return [list(row) for row in zip(*a)]
```

**Pure Python Solution (manual loops)**:
```python
def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    rows = len(a)
    cols = len(a[0])

    result = []
    for j in range(cols):
        new_row = []
        for i in range(rows):
            new_row.append(a[i][j])
        result.append(new_row)
    return result
```

**PyTorch Solution**:
```python
import torch

def transpose_matrix(a) -> torch.Tensor:
    a_t = torch.tensor(a, dtype=torch.float)
    return a_t.T  # or torch.transpose(a_t, 0, 1)
```

**Learned**:
- `zip(*a)` unpacks list and pairs elements at same indices
- `*` operator unpacks iterables (e.g., `*[[1,2], [3,4]]` → `[1,2], [3,4]`)
- PyTorch: `.T` property for transpose

---

### 3. Reshape Matrix (Easy - Linear Algebra)

**Problem**: Reshape a matrix to a new shape. Return `[]` if dimensions incompatible.

**Key Points**:
- Total elements must match: `rows × cols == new_rows × new_cols`
- Flatten first, then fill new shape row by row
- Check empty input BEFORE calculating size

**Pure Python Solution (manual)**:
```python
def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
    if not a or not a[0]:
        return []

    if len(a) * len(a[0]) != new_shape[0] * new_shape[1]:
        return []

    # Flatten
    flat = []
    for row in a:
        for val in row:
            flat.append(val)

    # Reshape
    result = []
    index = 0
    for i in range(new_shape[0]):
        new_row = []
        for j in range(new_shape[1]):
            new_row.append(flat[index])
            index += 1
        result.append(new_row)

    return result
```

**NumPy Solution**:
```python
import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
    if not a or not a[0]:
        return []

    arr = np.array(a)

    if arr.size != new_shape[0] * new_shape[1]:
        return []

    return arr.reshape(new_shape).tolist()
```

**PyTorch Solution**:
```python
import torch

def reshape_matrix(a, new_shape) -> torch.Tensor:
    a_t = torch.tensor(a, dtype=torch.float)

    if a_t.numel() != new_shape[0] * new_shape[1]:
        return []

    return a_t.reshape(new_shape).tolist()
```

**Mistakes Made**:
- ❌ `np.zeros(rows, cols)` - wrong syntax
- ✅ `np.zeros((rows, cols))` - shape must be a tuple
- ❌ Calculating size before checking if input is empty
- ❌ Returning `-1` instead of `[]` as specified
- ❌ Forgetting `.tolist()` to convert back to Python list

**Learned**:
- `np.zeros()` and `torch.zeros()` take shape as a **tuple**
- Always check empty input FIRST before accessing elements
- `arr.reshape(new_shape)` handles flatten + reshape in one step
- `.tolist()` converts numpy/torch arrays back to Python lists
- `arr.size` (NumPy) and `tensor.numel()` (PyTorch) give total element count

---

### 4. Calculate Mean by Row or Column (Easy - Linear Algebra)

**Problem**: Calculate mean of matrix by row or column based on mode parameter.

**Key Points**:
- Row mean: sum each row, divide by number of columns
- Column mean: sum each column, divide by number of rows
- Use `zip(*matrix)` trick to easily iterate columns

**Pure Python Solution**:
```python
def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    if not matrix or not matrix[0]:
        return []

    means = []

    if mode == 'column':
        for i in range(len(matrix[0])):
            col_sum = sum(matrix[j][i] for j in range(len(matrix)))
            means.append(col_sum / len(matrix))
    elif mode == 'row':  # Note: elif, not else if!
        for row in matrix:
            means.append(sum(row) / len(row))

    return means
```

**Pure Python Solution (using zip for columns)**:
```python
def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    if not matrix or not matrix[0]:
        return []

    if mode == 'column':
        # zip(*matrix) transposes - each column becomes a tuple
        return [sum(col) / len(col) for col in zip(*matrix)]
    elif mode == 'row':
        return [sum(row) / len(row) for row in matrix]

    return []
```

**NumPy Solution**:
```python
import numpy as np

def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    if not matrix or not matrix[0]:
        return []

    arr = np.array(matrix)

    if mode == 'column':
        return arr.mean(axis=0).tolist()  # axis=0 → along rows (collapse rows)
    elif mode == 'row':
        return arr.mean(axis=1).tolist()  # axis=1 → along cols (collapse cols)

    return []
```

**PyTorch Solution**:
```python
import torch

def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    if not matrix or not matrix[0]:
        return []

    t = torch.tensor(matrix, dtype=torch.float)

    if mode == 'column':
        return t.mean(dim=0).tolist()
    elif mode == 'row':
        return t.mean(dim=1).tolist()

    return []
```

**Mistakes Made**:
- ❌ `else if` - Python uses `elif`
- ✅ `elif` - correct Python syntax

**Learned**:
- Python uses `elif`, not `else if`
- `zip(*matrix)` gives columns as tuples (transpose trick)
- NumPy/PyTorch `axis`/`dim` parameter:
  - `axis=0` / `dim=0`: operate along rows (result has one value per column)
  - `axis=1` / `dim=1`: operate along columns (result has one value per row)

---

### 5. Scalar Multiplication of a Matrix (Easy - Linear Algebra)

**Problem**: Multiply every element in a matrix by a scalar value.

**Key Points**:
- Simple element-wise operation
- Use list comprehension for clean Python code
- NumPy/PyTorch: just use `*` operator directly on array/tensor

**Pure Python Solution**:
```python
def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
    return [[element * scalar for element in row] for row in matrix]
```

**Pure Python Solution (manual loops)**:
```python
def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
    result = []
    for row in matrix:
        new_row = []
        for element in row:
            new_row.append(element * scalar)
        result.append(new_row)
    return result
```

**NumPy Solution**:
```python
import numpy as np

def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
    arr = np.array(matrix)
    return (arr * scalar).tolist()
```

**PyTorch Solution**:
```python
import torch

def scalar_multiply(matrix, scalar) -> torch.Tensor:
    m_t = torch.as_tensor(matrix, dtype=torch.float)
    return m_t * scalar  # Returns tensor, use .tolist() for list
```

**Learned**:
- List comprehension: `[[expr for x in row] for row in matrix]` for 2D operations
- NumPy/PyTorch broadcast scalar automatically: `array * 2` multiplies all elements
- This is why arrays are preferred over lists - no loops needed!

**List vs Array comparison**:
```python
# List - need loops/comprehension
[[x * 2 for x in row] for row in matrix]

# Array - just multiply directly
array * 2  # Broadcasting handles it!
```

---

### 6. 2x2 Matrix Eigenvalues (Easy - Linear Algebra)

**Problem**: Calculate eigenvalues of a 2x2 matrix, sorted highest to lowest.

**Key Points**:
- Eigenvalues satisfy: det(A - λI) = 0
- For 2x2: λ² - trace(A)·λ + det(A) = 0
- Use quadratic formula to solve
- `linalg` = **lin**ear **alg**ebra submodule

**The Math**:
```
For matrix [[a, b], [c, d]]:

trace = a + d           (sum of diagonal)
det = ad - bc           (determinant)

λ = (trace ± √(trace² - 4·det)) / 2
```

**Pure Python Solution**:
```python
import math

def eigenvalues_2x2(matrix: list[list[float]]) -> list[float]:
    a, b = matrix[0]
    c, d = matrix[1]

    trace = a + d
    det = a * d - b * c

    discriminant = trace ** 2 - 4 * det
    sqrt_disc = math.sqrt(discriminant)

    lambda1 = (trace + sqrt_disc) / 2
    lambda2 = (trace - sqrt_disc) / 2

    return sorted([lambda1, lambda2], reverse=True)
```

**NumPy Solution**:
```python
import numpy as np

def eigenvalues_2x2(matrix: list[list[float]]) -> list[float]:
    eigenvalues = np.linalg.eigvals(matrix)
    return sorted(eigenvalues.tolist(), reverse=True)
```

**PyTorch Solution**:
```python
import torch

def eigenvalues_2x2(matrix: list[list[float]]) -> list[float]:
    m = torch.tensor(matrix, dtype=torch.float)
    eigenvalues = torch.linalg.eigvals(m).real  # Get real part
    return sorted(eigenvalues.tolist(), reverse=True)
```

**Learned**:
- `linalg` = linear algebra submodule (`np.linalg`, `torch.linalg`)
- `sorted(list, reverse=True)` for descending order (highest to lowest)
- Eigenvalue formula for 2x2: quadratic equation from trace and determinant
- Common `linalg` functions:
  ```python
  np.linalg.eigvals()   # Eigenvalues
  np.linalg.det()       # Determinant
  np.linalg.inv()       # Matrix inverse
  np.linalg.norm()      # Vector/matrix norm
  ```

---

### 7. Matrix Transformation T⁻¹AS (Easy - Linear Algebra)

**Problem**: Transform matrix A using T⁻¹AS where T and S are invertible matrices. Return -1 if not invertible.

**Key Points**:
- A matrix is invertible if det(M) ≠ 0
- Compute: T⁻¹ @ A @ S (matrix multiplication order matters!)
- Use `np.linalg.inv()` or `torch.linalg.inv()` for inverse

**Pure Python Solution (2x2 only)**:
```python
def transform_matrix(A: list[list[float]], T: list[list[float]], S: list[list[float]]) -> list[list[float]]:
    # Determinant of 2x2: ad - bc
    det_T = T[0][0] * T[1][1] - T[0][1] * T[1][0]
    det_S = S[0][0] * S[1][1] - S[0][1] * S[1][0]

    if det_T == 0 or det_S == 0:
        return -1

    # Inverse of 2x2: [[a,b],[c,d]]⁻¹ = (1/det) * [[d,-b],[-c,a]]
    T_inv = [
        [T[1][1] / det_T, -T[0][1] / det_T],
        [-T[1][0] / det_T, T[0][0] / det_T]
    ]

    # Helper function for 2x2 matrix multiplication
    def mat_mul(X, Y):
        return [
            [X[0][0]*Y[0][0] + X[0][1]*Y[1][0], X[0][0]*Y[0][1] + X[0][1]*Y[1][1]],
            [X[1][0]*Y[0][0] + X[1][1]*Y[1][0], X[1][0]*Y[0][1] + X[1][1]*Y[1][1]]
        ]

    return mat_mul(mat_mul(T_inv, A), S)
```

**NumPy Solution**:
```python
import numpy as np

def transform_matrix(A, T, S):
    A = np.array(A)
    T = np.array(T)
    S = np.array(S)

    if np.linalg.det(T) == 0 or np.linalg.det(S) == 0:
        return -1

    T_inv = np.linalg.inv(T)
    result = T_inv @ A @ S

    return result.tolist()
```

**PyTorch Solution**:
```python
import torch

def transform_matrix(A, T, S) -> torch.Tensor:
    A_t = torch.as_tensor(A, dtype=torch.float)
    T_t = torch.as_tensor(T, dtype=torch.float)
    S_t = torch.as_tensor(S, dtype=torch.float)

    if torch.linalg.det(T_t) == 0 or torch.linalg.det(S_t) == 0:
        return torch.tensor(-1.)

    T_inv = torch.linalg.inv(T_t)
    result = T_inv @ A_t @ S_t

    # Round to 3 decimals
    return torch.round(result * 1000) / 1000
```

**Learned**:
- Matrix is invertible ↔ det(M) ≠ 0
- `np.linalg.inv()` / `torch.linalg.inv()` computes matrix inverse
- 2x2 inverse formula: `[[a,b],[c,d]]⁻¹ = (1/det) * [[d,-b],[-c,a]]`
- Matrix multiplication order matters: T⁻¹ @ A @ S ≠ S @ A @ T⁻¹
- Rounding trick: `torch.round(x * 1000) / 1000` for 3 decimal places

---

### 8. 2x2 Matrix Inverse (Medium - Linear Algebra)

**Problem**: Calculate the inverse of a 2x2 matrix. Return `None` if not invertible.

**Key Points**:
- Inverse exists only if det ≠ 0
- Formula: `A⁻¹ = (1/det) * [[d, -b], [-c, a]]`
- Verify: A @ A⁻¹ = Identity matrix

**The Formula**:
```
For [[a, b], [c, d]]:

det = ad - bc

Inverse = (1/det) * [[d, -b],
                     [-c, a]]

Note: Swap a↔d, negate b and c, divide by det
```

**Pure Python Solution**:
```python
def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:
    a, b = matrix[0]
    c, d = matrix[1]

    det = a * d - b * c

    if det == 0:
        return None

    return [
        [d / det, -b / det],
        [-c / det, a / det]
    ]
```

**NumPy Solution**:
```python
import numpy as np

def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:
    arr = np.array(matrix)

    if np.linalg.det(arr) == 0:
        return None

    return np.linalg.inv(arr).tolist()
```

**PyTorch Solution**:
```python
import torch

def inverse_2x2(matrix) -> torch.Tensor | None:
    m = torch.as_tensor(matrix, dtype=torch.float)

    if torch.linalg.det(m) == 0:
        return None

    return torch.linalg.inv(m)
```

**Example Walkthrough**:
```
matrix = [[4, 7], [2, 6]]

a=4, b=7, c=2, d=6
det = 4*6 - 7*2 = 24 - 14 = 10

Inverse = (1/10) * [[6, -7], [-2, 4]]
        = [[0.6, -0.7], [-0.2, 0.4]]

Verify: [[4,7],[2,6]] @ [[0.6,-0.7],[-0.2,0.4]] = [[1,0],[0,1]] ✓
```

**Learned**:
- 2x2 inverse formula: swap diagonal, negate off-diagonal, divide by det
- `| None` in type hints means function can return `None`
- Always check `det ≠ 0` before computing inverse
- This formula only works for 2x2; use `np.linalg.inv()` for larger matrices

---

### 9. Matrix Multiplication (Medium - Linear Algebra)

**Problem**: Multiply two matrices A × B. Return -1 if dimensions don't align.

**Key Points**:
- A (m × n) × B (n × p) = C (m × p)
- **Requirement**: A.columns == B.rows (`len(a[0]) == len(b)`)
- Each element C[i][j] = dot product of row i of A with column j of B

**Visual**:
```
A (2×3)         B (3×2)         C (2×2)
[a b c]    ×    [g h]      =    [ag+bi+ck  ah+bj+cl]
[d e f]         [i j]           [dg+ei+fk  dh+ej+fl]
                [k l]
```

**Pure Python Solution**:
```python
def matrixmul(a: list[list[int|float]], b: list[list[int|float]]) -> list[list[int|float]]:
    # Check: A columns must equal B rows
    if len(a[0]) != len(b):
        return -1

    result = []
    for i in range(len(a)):          # Each row of A
        row = []
        for j in range(len(b[0])):   # Each column of B
            # Dot product: row i of A with column j of B
            total = sum(a[i][k] * b[k][j] for k in range(len(b)))
            row.append(total)
        result.append(row)

    return result
```

**NumPy Solution**:
```python
import numpy as np

def matrixmul(a, b):
    a = np.array(a)
    b = np.array(b)

    if a.shape[1] != b.shape[0]:
        return -1

    return (a @ b).tolist()
```

**PyTorch Solution**:
```python
import torch

def matrixmul(a, b) -> torch.Tensor:
    a_t = torch.as_tensor(a)
    b_t = torch.as_tensor(b)

    if a_t.size(1) != b_t.size(0):
        return torch.tensor(-1)

    return a_t @ b_t  # or a_t.matmul(b_t)
```

**Example Walkthrough**:
```
A = [[1, 2],      B = [[2, 1],
     [2, 4]]           [3, 4]]

A is 2×2, B is 2×2 → Valid! Result is 2×2

C[0][0] = 1*2 + 2*3 = 2 + 6 = 8
C[0][1] = 1*1 + 2*4 = 1 + 8 = 9
C[1][0] = 2*2 + 4*3 = 4 + 12 = 16
C[1][1] = 2*1 + 4*4 = 2 + 16 = 18

Result = [[8, 9], [16, 18]]
```

**Learned**:
- Dimension rule: (m×n) @ (n×p) = (m×p) — inner dimensions must match
- `@` operator works for both NumPy and PyTorch
- Alternative methods: `np.dot()`, `np.matmul()`, `torch.matmul()`, `torch.mm()`
- Matrix multiplication is NOT commutative: A @ B ≠ B @ A

---

### 10. Covariance Matrix (Medium - Linear Algebra)

**Problem**: Calculate the covariance matrix for a set of feature vectors.

**Key Points**:
- Input: list of features, each feature has n observations
- Covariance measures how two variables change together
- Diagonal = variance, Off-diagonal = covariance between pairs
- Formula: `Cov(X,Y) = Σ(Xi - X̄)(Yi - Ȳ) / (n-1)`

**The Concept**:
```
Covariance matrix is symmetric:
        Feature1  Feature2
Feat1 [ Var(1)    Cov(1,2) ]
Feat2 [ Cov(2,1)  Var(2)   ]

Cov(X,Y) > 0: X↑ → Y↑ (positive correlation)
Cov(X,Y) < 0: X↑ → Y↓ (negative correlation)
Cov(X,Y) = 0: no linear relationship
```

**Pure Python Solution**:
```python
def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    n_features = len(vectors)
    n_observations = len(vectors[0])

    # Calculate means
    means = [sum(feature) / n_observations for feature in vectors]

    # Build covariance matrix
    cov_matrix = []
    for i in range(n_features):
        row = []
        for j in range(n_features):
            cov = sum(
                (vectors[i][k] - means[i]) * (vectors[j][k] - means[j])
                for k in range(n_observations)
            ) / (n_observations - 1)
            row.append(cov)
        cov_matrix.append(row)

    return cov_matrix
```

**NumPy Solution**:
```python
import numpy as np

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    return np.cov(vectors).tolist()
```

**PyTorch Solution**:
```python
import torch

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    t = torch.tensor(vectors, dtype=torch.float)
    # Center the data
    t_centered = t - t.mean(dim=1, keepdim=True)
    n = t.size(1)
    # Cov = X_centered @ X_centered.T / (n-1)
    cov = (t_centered @ t_centered.T) / (n - 1)
    return cov.tolist()
```

**Example Walkthrough**:
```
vectors = [[1, 2, 3], [4, 5, 6]]

means = [2, 5]

Deviations:
Feature 1: [1-2, 2-2, 3-2] = [-1, 0, 1]
Feature 2: [4-5, 5-5, 6-5] = [-1, 0, 1]

Cov(1,1) = (1 + 0 + 1) / 2 = 1.0  (variance of feature 1)
Cov(1,2) = (1 + 0 + 1) / 2 = 1.0
Cov(2,2) = (1 + 0 + 1) / 2 = 1.0  (variance of feature 2)

Result: [[1.0, 1.0], [1.0, 1.0]]
```

**Learned**:
- Covariance formula: `Σ(Xi - X̄)(Yi - Ȳ) / (n-1)`
- Division by `n-1` (not n) = "Bessel's correction" for sample covariance
- `np.cov()` expects features as rows (each row = one variable)
- Covariance matrix is always symmetric
- Matrix form: `Cov = X_centered @ X_centered.T / (n-1)`

---

### 11. Jacobi Method (Medium - Linear Algebra)

**Problem**: Solve system of linear equations Ax = b using Jacobi iterative method.

**Key Points**:
- Iterative method (doesn't solve directly like matrix inversion)
- Uses OLD values to compute NEW values (unlike Gauss-Seidel)
- Formula: `x[i] = (b[i] - Σ(A[i][j]*x[j] for j≠i)) / A[i][i]`
- Converges if matrix is diagonally dominant

**The Algorithm**:
```
1. Start with initial guess x = [0, 0, ..., 0]
2. For each iteration:
   For each equation i:
     x_new[i] = (b[i] - sum of off-diagonal terms) / diagonal
3. Replace x with x_new
4. Repeat n times
```

**Pure Python Solution**:
```python
def solve_jacobi(A: list[list[float]], b: list[float], n: int) -> list[float]:
    size = len(b)
    x = [0.0] * size  # Initial guess

    for _ in range(n):
        x_new = [0.0] * size
        for i in range(size):
            # Sum of A[i][j] * x[j] for j != i
            total = sum(A[i][j] * x[j] for j in range(size) if j != i)
            x_new[i] = round((b[i] - total) / A[i][i], 4)
        x = x_new

    return x
```

**NumPy Solution**:
```python
import numpy as np

def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    size = len(b)
    x = np.zeros(size)

    for _ in range(n):
        x_new = np.zeros(size)
        for i in range(size):
            total = np.dot(A[i], x) - A[i][i] * x[i]  # All except diagonal
            x_new[i] = (b[i] - total) / A[i][i]
        x = np.round(x_new, 4)

    return x.tolist()
```

**NumPy Solution (Vectorized)**:
```python
import numpy as np

def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    D = np.diag(A)          # Diagonal elements [a11, a22, a33, ...]
    R = A - np.diag(D)      # Off-diagonal matrix (zeros on diagonal)
    x = np.zeros(len(b))

    for _ in range(n):
        x = (b - R @ x) / D
        x = np.round(x, 4)

    return x.tolist()
```

**Example Walkthrough**:
```
A = [[5, -2, 3],      b = [-1, 2, 3],  n = 2
     [-3, 9, 1],
     [2, -1, -7]]

Iteration 1 (x = [0, 0, 0]):
x[0] = (-1 - (-2*0 + 3*0)) / 5 = -0.2
x[1] = (2 - (-3*0 + 1*0)) / 9 = 0.2222
x[2] = (3 - (2*0 - 1*0)) / -7 = -0.4286

Iteration 2 (x = [-0.2, 0.2222, -0.4286]):
x[0] = (-1 - (-2*0.2222 + 3*-0.4286)) / 5 = 0.146
x[1] = (2 - (-3*-0.2 + 1*-0.4286)) / 9 = 0.2032
x[2] = (3 - (2*-0.2 - 1*0.2222)) / -7 = -0.5175

Result: [0.146, 0.2032, -0.5175]
```

**Learned**:
- Jacobi uses OLD x values to compute ALL new values simultaneously
- `np.diag(A)` extracts diagonal elements as 1D array
- `np.diag(vector)` creates diagonal matrix from 1D array
- Vectorized formula: `x_new = (b - R @ x) / D`
- Converges for diagonally dominant matrices (|a_ii| > Σ|a_ij| for j≠i)

**Jacobi vs Gauss-Seidel**:
```
Jacobi:       Uses old x values for all computations
Gauss-Seidel: Uses updated x values immediately (faster convergence)
```

---

## Quick Reference

### Tensor Creation (PyTorch)
```python
torch.tensor([1, 2, 3])              # From list
torch.as_tensor(data, dtype=torch.float)  # Flexible input
torch.zeros(3, 4)                    # 3x4 zeros
torch.ones(2, 2)                     # 2x2 ones
```

### Tensor Operations
```python
a @ b                    # Matrix multiplication
a * b                    # Element-wise multiplication
a.shape                  # Dimensions
a.size(0)                # Size of dimension 0
a.tolist()               # Convert to Python list
```

### Dimension Checking
```python
# Pure Python
len(matrix[0]) != len(vector)

# PyTorch
a_t.size(1) != b_t.size(0)
```

---

## Notes for Future Problems

<!-- Add new problems below this line -->

