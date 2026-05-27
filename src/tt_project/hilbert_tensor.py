import numpy as np

def hilbert_tensor(shape):
    d = len(shape)
    idx = np.indices(shape)# gave index of each dimension
    H = 1.0 / (idx.sum(axis=0) + 1)
    return H