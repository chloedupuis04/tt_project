

import numpy as np
from tensorly.contrib.decomposition import tensor_train_cross
from tt_project.tt_svd_delta import tt_svd_delta
from tt_project.right_ortho import right_ortho
from tt_project.tt_rounding import tt_rounding
from tt_project.tt_to_tensor import tt_to_tensor


class TT:
    def __init__(self, cores):
        self.cores= [np.array(core) for core in cores]

    def from_TTSVD(tensor,eps):
        cores, cores_shape, ranks = tt_svd_delta(tensor, eps)
        return TT(cores)
    
    def from_TTcross(tensor,max_rank,eps,d_theta):
        max_ranks=[1]+[max_rank]*(d_theta-1)+[1]
        cores = tensor_train_cross(tensor, max_ranks,eps)
        return TT(cores)

    def random_initialization(shape,ranks,):
        cores = []
        d = len(shape)
        for i in range(d):
            if i == 0:
                core = np.random.rand(1, shape[i], ranks[i])
            elif i == d - 1:
                core = np.random.rand(ranks[i-1], shape[i], 1)
            else:
                core = np.random.rand(ranks[i-1], shape[i], ranks[i])
            cores.append(core)
        return TT(cores)

    def orthogonalization_right(self):
        self.cores= right_ortho(self.cores)
        return self
    
    def tt_to_tensor(self):
        return tt_to_tensor(self.cores)
    
    def rounding(self, eps):
        self.cores, ranks = tt_rounding(self.cores, eps)
        return self, ranks
    
    
    def total_entries(self):
        return sum(G.size for G in self.cores)
    
    def get_tt_ranks(self):
        ranks = [self.cores[0].shape[0]]
        for G in self.cores:
            ranks.append(G.shape[2])
        return ranks
    