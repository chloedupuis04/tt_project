import numpy as np
from tt_project.tt_to_tensor import tt_to_tensor

def err(A,cores):
    B = tt_to_tensor(cores)
    error = np.linalg.norm(A - B)
    rel_err = error / np.linalg.norm(A)
    print("the error is :"+str(error))
    print("the relative error is: " +str(rel_err))