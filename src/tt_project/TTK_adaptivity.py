from tt_project.greedy2_cross import *
from tt_project.adaptivity_in_n import *
from tt_project.vertical_core import vertical_core
from tt_project.horizontal_core import horizontal_core
from tt_project.tt_rounding import tt_rounding
from tt_project.tt_to_tensor import tt_to_tensor
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

def find_n_list(n,domain,nbr_samples,f):
    list_n=[]
    list_intervals=[]
    d=domain.shape[0]
    for axis in range(d):
        root=IntervalTree(domain[axis, 0], domain[axis, 1])
        find_subinterval_d(domain,n,axis,nbr_samples,f,root,depth=0,max_depth=100,tol=1e-6)
        intervals=root.intervals()
        list_intervals.append(intervals)
        list_n.append(n*len(intervals))
    return list_n,list_intervals

def phi(alpha, beta, x):
    return (beta - alpha) / 2 * x + (alpha + beta) / 2



def Phi(x, eta, a, b, n, atol=1e-14):

    N = n - 1

    is_endpoint = (
        np.isclose(eta, a, atol=atol, rtol=0.0)
        or np.isclose(eta, b, atol=atol, rtol=0.0)
    )

    c_eta = 2.0 if is_endpoint else 1.0

    result = 0.5 * np.ones_like(np.asarray(x, dtype=float))

    for k in range(1, N):
        Tk_x = cheb.Chebyshev.basis(k, domain=[a, b])(x)
        Tk_eta = cheb.Chebyshev.basis(k, domain=[a, b])(eta)
        result += Tk_x * Tk_eta

    TN_x = cheb.Chebyshev.basis(N, domain=[a, b])(x)
    TN_eta = cheb.Chebyshev.basis(N, domain=[a, b])(eta)

    result += 0.5 * TN_x * TN_eta

    result *= 2.0 / (N * c_eta)

    return float(result) if np.ndim(result) == 0 else result


def khatri_rao(A,B):
    m,q1=A.shape
    p,q2=B.shape
    if q1!=q2:
        raise ValueError("number of columns must be the same")
    return np.stack([np.kron(A[:,i],B[:,i]) for i in range(q1)],axis=1)#stack the kronecker products of the columns of A and B along the second axis

def concatenated_nodes(intervals, n):
    nodes = []
    for a, b in intervals:
        nodes.append(phi(a, b, cheb.chebpts2(n)))

    return np.concatenate(nodes)



def TTK_adaptivity_offline(n,domain,nbr_samples,f):

    list_n,list_intervals=find_n_list(n,domain,nbr_samples,f)
    discretized_nodes = [concatenated_nodes(list_intervals[j], n)for j in range(len(list_intervals))]

    def func_aux(ind):
        point = [discretized_nodes[j][ind[j]] for j in range(len(ind))]
        return f(*point)
    tol=1e-6 
    tt_cores, Jyl, Jyr, ilocl, ilocr, evalcnt = greedy2_cross(list_n, func_aux, tol = tol)
    print("Shape of tt cores: ", [tt_core.shape for tt_core in tt_cores])
    return tt_cores,list_n, list_intervals


def loc_xj(intervals, x, atol=1e-14):
    ell = np.searchsorted(intervals[:, 1], x, side="left")
    ell = min(ell, len(intervals) - 1)

    if not (intervals[ell, 0] - atol<= x<= intervals[ell, 1] + atol):
        raise ValueError(f"No interval containing x={x} was found.")

    return ell


def adaptive_q(x, intervals, n, Phi):

    intervals = np.asarray(intervals, dtype=float)

    m_i = len(intervals)
    q = np.zeros(m_i * n)

    # Active interval
    ell = loc_xj(intervals, x)
    a, b = intervals[ell]

    # Local Chebyshev nodes in [a, b]
    reference_nodes = cheb.chebpts2(n)
    local_nodes = phi(a, b, reference_nodes)

    # Local interpolation vector
    q_local = np.array([Phi(x, node, a, b, n) for node in local_nodes],dtype=float,)

    # Global index k_i = ell*n + r
    start = ell * n
    stop = (ell + 1) * n
    q[start:stop] = q_local

    return q


def TTK_adaptivity_online(n,domain,tt_cores,list_n,list_intervals,x_points,y_points,d,N_s,N_t,tol):
    D=domain.shape[0]
      
    #def of Ui,Vi
    U=[]
    V=[]
    x_intervals=list_intervals[:d]
    y_intervals=list_intervals[d:]
    
    for i in range(d):
        U_i=np.zeros((N_s,list_n[0]))
        V_i=np.zeros((N_t,list_n[1]))

        for k in range(N_s):
            U_i[k,:]=adaptive_q(x_points[i,k],x_intervals[i],n,Phi)
            
        for k in range(N_t):
            V_i[k,:]=adaptive_q(y_points[i,k],y_intervals[i],n,Phi)

        U.append(U_i)
        V.append(V_i)


    #phase3: construct factor matrix 
    S=U[0]@vertical_core(tt_cores[0])
    for i in range(1,d):
        G_i2=vertical_core(tt_cores[i])
        S = khatri_rao(S.T, (U[i]).T).T@G_i2

    T=horizontal_core(tt_cores[D-1])@V[d-1].T
    for i in range(1,d):
        T=horizontal_core(tt_cores[D-1-i])@khatri_rao(V[d-1-i].T, T)
    T=T.T

    #phase4 
    S_twisted=S[None,:,:]
    T_twisted=T.T[:,:,None]

    cores_approx=[S_twisted,T_twisted]

    tt_afterrounding,ranks_rounding=tt_rounding(cores_approx)
    print("Shape of tt cores: ", [tt_after.shape for tt_after in tt_afterrounding])
    K=tt_to_tensor(tt_afterrounding)

    return K

