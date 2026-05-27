import numpy as np

def sqrt_tensor(shape, a, b):
    idx = np.indices(shape)
    total = np.zeros(shape, dtype=float)

    for j, n in enumerate(shape):
        if n == 1:
            total += a
        else:
            total += ((n - 1 - idx[j]) / (n - 1)) * a + (idx[j] / (n - 1)) * b

    return np.sqrt(total)