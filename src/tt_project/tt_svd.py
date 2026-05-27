
import numpy as np

def tt_svd(A,ranks):
    #initialization
    dims=A.shape
    d=len(dims)
    cores=[]
    cores_shape=[]
    C=A.copy() #copy the input tensor 
    for k in range (1,d):
        r_prev=ranks[k-1]
        r_k=ranks[k]
        n_k=dims[k-1]
        C=C.reshape(r_prev*n_k,-1)
        U,S,Vt=np.linalg.svd(C,full_matrices=False)
        # Truncate the SVD to the specified rank r_k
        U_k= U[:,:r_k]
        S_k= S[:r_k]
        Vt_k= Vt[:r_k,:]
        G_k=U_k.reshape(r_prev,n_k,r_k) # Reshape U_k to form the k-th core tensor G_k
        cores.append(G_k)
        cores_shape.append(G_k.shape)
        C=np.diag(S_k)@Vt_k # Update C for the next iteration
    C=C.reshape(ranks[-2],dims[-1],ranks[-1])
    cores.append(C)
    cores_shape.append(C.shape)
    
    return cores,cores_shape,ranks