import numpy as np
from tt_project.delta_rank import delta_rank 

def tt_svd_delta(A,eps=1e-6):
    #initialization
    dims=A.shape
    d=len(dims)
    cores=[]
    cores_shape=[]
    C=A.copy()
    ranks=[]
    ranks.append(1)
    delta=(eps/np.sqrt(d-1))*np.linalg.norm(A)

    for k in range (1,d):
        r_prev=ranks[k-1]
        n_k=dims[k-1]
        C=C.reshape(r_prev*n_k,-1)
        U,S,Vt=np.linalg.svd(C,full_matrices=False)
        r_delta=delta_rank(S,delta)
        ranks.append(r_delta)
        # Truncate the SVD to the specified rank r_delta
        U_k= U[:,:r_delta]
        S_k= S[:r_delta]
        Vt_k= Vt[:r_delta,:]
        G_k=U_k.reshape(r_prev,n_k,r_delta)# Reshape U_k to form the k-th core tensor G_k
        cores.append(G_k)
        cores_shape.append(G_k.shape)
        C=np.diag(S_k)@Vt_k# Update C for the next iteration
    C=C.reshape(ranks[-1],dims[-1],1)
    cores.append(C)
    cores_shape.append(C.shape)
    ranks.append(1)
    
    return cores,cores_shape,ranks