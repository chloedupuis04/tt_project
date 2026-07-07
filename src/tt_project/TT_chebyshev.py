import numpy as np
import time 
from numpy.polynomial import chebyshev as cheb
from tt_project.vertical_core import vertical_core
from tt_project.horizontal_core import horizontal_core
from tt_project.TT import TT

def phi(x,alpha,beta):
    return (beta-alpha)/2*x + (alpha+beta)/2

def Phi(x, eta, a, b, n):
    s = 0.0
    for k in range(1, n):
        Tk_x = cheb.Chebyshev.basis(k, domain=[a, b])(x)
        Tk_eta = cheb.Chebyshev.basis(k, domain=[a, b])(eta)
        s += Tk_x * Tk_eta
    return (2/n) * s + (1/n)


def TT_Chebyshev_interpolation(f,domain,n,d_theta,tol=1e-6,tt_cross=False):
    t0=time.perf_counter()

    #phase1:build M
    Eta = []
    cheb_nodes = cheb.chebpts1(n)
    for j in range(d_theta):
        eta_j = phi(cheb_nodes, domain[j, 0], domain[j, 1])
        Eta.append(eta_j)
    M=np.zeros(tuple([n]*d_theta))

    for idx in np.ndindex(*M.shape): 
        eta_point=[Eta[j][idx[j]] for j in range(d_theta)]
        M[idx]=f(eta_point)
    t1=time.perf_counter()
    print("time for stage 1 is :"+str(t1-t0)+" seconds")

    #phase2:compute the approximation

    #cores,cores_shape,ranks=tt_svd_delta(M,eps=tol)
    if tt_cross:
        raise NotImplementedError("Don't know how cna i use this as we need ranks as input for the tt_cross")
    else:
        tt=TT.from_TTSVD(M,tol)
    #err(M,cores) if we want to know the error of the tt approximation of M
    t2=time.perf_counter()
    print("time for stage 2 is :"+str(t2-t1)+" seconds")

    cores_afterrounding,ranks_rounding=tt.rounding(eps=tol)
    t3=time.perf_counter()
    print("time for rounding is :"+str(t3-t2)+" seconds")
    print("-------------------")
    print("total time is :"+str(t3-t0)+" seconds")

    return cores_afterrounding, ranks_rounding
   
def TT_Chebyshev_evaluation(theta, d_theta,domain,n,cores_afterrounding,ranks_rounding):
    cheb_nodes = cheb.chebpts1(n)
    H=np.zeros((ranks_rounding[0],ranks_rounding[d_theta-1]))
    for j in range (d_theta):
        qp_j=[Phi(theta[j],phi(cheb_node,domain[j,0],domain[j,1]),domain[j,0],domain[j,1],n) for cheb_node in cheb_nodes]
        H_j=np.tensordot(cores_afterrounding.cores[j],qp_j,axes=([1],[0]))#bc we sum over the second dimension of the core and the first dimension of qp_j
        if j==0:
            H=H_j
        else:
            H=H@H_j
    return H

