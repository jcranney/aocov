Efficient covariance computations for Adaptive Optics in Python

## Introduction
In Adaptive Optics control/estimation/simulations, it's common to find yourself performing an operation similar to:
```python
import numpy as np
import aotools

# define some grid coordinates in x and y
n = 64
xx, yy = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
xx = xx.flatten()
yy = yy.flatten()

# compute the distances between every pair of coordinates in this grid
rr = ((xx[:, None]-xx[None, :])**2 + (yy[:, None]-yy[None, :])**2)**0.5

# evaluate the covariance at all points in 
r0 = 0.15
L0 = 25.0
cov = aotools.phase_covariance(rr, r0, L0)
```

The goal of this package is to address 2 main limitations of this simple operation:
 1) The evaluation of the covariance function is highly parallelisable, though the implementation above is limited to CPU execution.
 2) The distance matrix (`rr` in this case) is highly redundant, with many repeated values due to the regular geometry of the inputs.

These problems are addressed simultaneously by:
 1) using PyTorch to execute von Karman covariance functions on (e.g.) GPU, by simply defining a device to execute on. Note that PyTorch must be installed with the appropriate configuration, though for modern CUDA GPU devices, this is done by default. Just use `device="gpu"` when you call `aocov.phase_covariance`.
 2) calculating the unique elements of the distance matrix, and only evaluating the covariance function for these values, then copying the values for all other duplicated values.
 
Modifying the above example, we would instead use:
```python
import numpy as np
import aocov # <-- import this package

# define some grid coordinates in x and y
n = 64
xx, yy = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
xx = xx.flatten()
yy = yy.flatten()

# compute the distances between every pair of coordinates in this grid
rr = ((xx[:, None]-xx[None, :])**2 + (yy[:, None]-yy[None, :])**2)**0.5

# evaluate a von Karman covariance on GPU
r0 = 0.15
L0 = 25.0
cov = aocov.phase_covariance(rr, r0, L0, device="cuda:0")
```

Furthermore, since the distance computations can become expensive, and is prone to typos, we provide a further convenience function, `phase_covariance_xyxy`. Finally, the example becomes:
```python
import numpy as np
import aocov # <-- import this package

# define some grid coordinates in x and y
n = 64
xx, yy = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
xx = xx.flatten()
yy = yy.flatten()

# evaluate a von Karman covariance on GPU
r0 = 0.15
L0 = 25.0
cov = aocov.phase_covariance_xyxy(xx, yy, xx, yy, r0, L0, device="cuda:0")
```

Note that the last option is likely to perform the fastest, see [performance comparison](#performance) below.

## Performance
