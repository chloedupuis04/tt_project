import numpy as np
from tt_project.horizontal_core import horizontal_core
from tt_project.vertical_core import vertical_core
from tt_project.v2h import v2h


def tt_partialcontractionRL(coresX,coresY):
    d=len(coresX)
    N=[core.shape[1] for core in coresX]#lost of the dimensions of the tensor
    W=[0 for _ in range(d)] #list that store the intermediate results of the partial contraction
    
    H_y=horizontal_core(coresY[-1])#define a function that compute the horizontal tensor of a 3 dimensional core to simplify the notation
    H_x=horizontal_core(coresX[-1])
    W[d-1]=H_x@H_y.T

    for k in range(d-1,0,-1):
        V_x=vertical_core(coresX[k-1])
        H_y=horizontal_core(coresY[k-1])
        V_z=V_x@W[k]
        H_z=v2h(V_z,N[k-1])
        W[k-1]=H_z@H_y.T


    return W