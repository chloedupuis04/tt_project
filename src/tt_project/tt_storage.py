import numpy as np

def tt_storage(cores):
    return sum(G.size for G in cores)

def compression_ratio(full_tensor, cores):
    full_size = np.prod(full_tensor.shape)
    tt_size = tt_storage(cores)
    return full_size/tt_size