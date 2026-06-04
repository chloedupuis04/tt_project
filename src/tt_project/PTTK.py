
import numpy as np
import time 
from numpy.polynomial import chebyshev as cheb
from tt_project.tt_svd_delta import tt_svd_delta
from tt_project.vertical_core import vertical_core
from tt_project.horizontal_core import horizontal_core
from tt_project.tt_rounding import tt_rounding
from tt_project.tt_storage import tt_storage, compression_ratio
from tt_project.err import err
from tt_project.tt_to_tensor import tt_to_tensor

def phi(x,alpha,beta):
    return (beta-alpha)/2*x + (alpha+beta)/2

def Phi(x, eta, a, b, n):
    s = 0.0
    for k in range(1, n):
        Tk_x = cheb.Chebyshev.basis(k, domain=[a, b])(x)
        Tk_eta = cheb.Chebyshev.basis(k, domain=[a, b])(eta)
        s += Tk_x * Tk_eta
    return (2/n) * s + (1/n)


def khatri_rao(A,B):
    m,q1=A.shape
    p,q2=B.shape
    if q1!=q2:
        raise ValueError("number of columns must be the same")
    return np.stack([np.kron(A[:,i],B[:,i]) for i in range(q1)],axis=1)#stack the kronecker products of the columns of A and B along the second axis

def PTTK_offline(f,domain,n,x_points,y_points,d,d_theta,N_s,N_t,tol=1e-6):
    t0=time.perf_counter()

    #phase1:build M and U_i,V_i
    D = domain.shape[0]


    Eta = []
    cheb_nodes = cheb.chebpts1(n)
    for j in range(D):
        eta_j = phi(cheb_nodes, domain[j, 0], domain[j, 1])
        Eta.append(eta_j)
    M=np.zeros(tuple([n]*D))


    for idx in np.ndindex(*M.shape): 
        eta_point=[Eta[j][idx[j]] for j in range(D)]
        x_eta=eta_point[:d]
        theta_eta=eta_point[d:d+d_theta]
        y_eta=eta_point[d+d_theta:]
        M[idx]=f(x_eta,theta_eta,y_eta)
    
    #def of Ui,Vi
    U=[]
    V=[]

    for i in range(d):
        U_i=np.zeros((N_s,n))
        V_i=np.zeros((N_t,n))

        for k in range(N_s):

            qs_i=[Phi(x_points[i,k],phi(cheb_node,domain[i,0],domain[i,1]),domain[i,0],domain[i,1],n) for cheb_node in cheb_nodes]
            U_i[k,:]=qs_i
            

        for k in range(N_t):
            qt_i=[Phi(y_points[i,k],phi(cheb_node,domain[i+d+d_theta,0],domain[i+d+d_theta,1]),domain[i+d+d_theta,0],domain[i+d+d_theta,1],n) for cheb_node in cheb_nodes]
            V_i[k,:]=qt_i

        U.append(U_i)
        V.append(V_i)

    t1=time.perf_counter()
    print("time for stage 1 is :"+str(t1-t0)+" seconds")


    #phase2:compute the approximation
    cores,cores_shape,ranks=tt_svd_delta(M,eps=tol)
    #err(M,cores) if we want to know the error of the tt approximation of M
    t2=time.perf_counter()
    print("time for stage 2 is :"+str(t2-t1)+" seconds")

    #phase3: construct factor matrix 
    S=U[0]@vertical_core(cores[0])
    for i in range(1,d):
        G_i2=vertical_core(cores[i])
        S = khatri_rao(S.T, (U[i]).T).T@G_i2

    T=horizontal_core(cores[D-1])@V[d-1].T
    for i in range(1,d):
        T=horizontal_core(cores[D-1-i])@khatri_rao(V[d-1-i].T, T)
    T=T.T

    t3=time.perf_counter()
    print("time for stage 3 is :"+str(t3-t2)+" seconds")

    #phase4 
    S_twisted=S[None,:,:]
    T_twisted=T.T[:,:,None]

    cores_approx=[S_twisted]

    for j in range(d_theta):
        cores_approx.append(cores[d+j])

    cores_approx.append(T_twisted)
    cores_afterrounding,ranks_rounding=tt_rounding(cores_approx,eps=1e-6)
    t4=time.perf_counter()
    print("time for stage 4 is :"+str(t4-t3)+" seconds")
    print("-------------------")
    print("total time is :"+str(t4-t0)+" seconds")

    return cores_afterrounding, ranks_rounding
   
def PTTK_online(theta, d_theta,d,domain,n,cores_afterrounding,ranks_rounding):
    cheb_nodes = cheb.chebpts1(n)
    H=np.zeros((ranks_rounding[1],ranks_rounding[1+d_theta]))
    for j in range (d_theta):
        qp_j=[Phi(theta[j],phi(cheb_node,domain[d+j,0],domain[d+j,1]),domain[d+j,0],domain[d+j,1],n) for cheb_node in cheb_nodes]
        H_j=np.tensordot(cores_afterrounding[j+1],qp_j,axes=([1],[0]))#bc we sum over the second dimension of the core and the first dimension of qp_j
        if j==0:
            H=H_j
        else:
            H=H@H_j

    K=cores_afterrounding[0].reshape((-1,ranks_rounding[1]))@H@cores_afterrounding[-1].reshape((ranks_rounding[-2],-1))

    return K

