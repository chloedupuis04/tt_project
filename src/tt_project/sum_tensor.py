import numpy as np
from tt_project.tt_to_tensor import tt_to_tensor

def sum_tensor(coresA,coresB):

    d=len(coresA)
    coresZ=[] #list to store the cores of the sum tensor
    ranks=[]

    for k in range(d):
        G_A=coresA[k]
        G_B=coresB[k]

        rA_prev,n_k,rA_k=G_A.shape
        rB_prev,nB_k,rB_k=G_B.shape

        if n_k != nB_k:
            raise ValueError(f"Mode size mismatch at core {k}: {n_k} != {nB_k}")

        if k==0:
            ranks.append(1)
            G_Z=np.zeros((1,n_k,rA_k+rB_k))
            G_Z[:,:,:rA_k]=G_A
            G_Z[:,:,rA_k:]=G_B
        
        elif k==d-1:
            rZ=rA_prev+rB_prev
            ranks.append(rZ)
            G_Z=np.zeros((rZ,n_k,1))
            G_Z[:rA_prev,:,:]=G_A
            G_Z[rA_prev:,:,:]=G_B
            
        else:
            rZ=rA_prev+rB_prev
            ranks.append(rZ)
            G_Z=np.zeros((rZ,n_k,rA_k+rB_k))
            G_Z[:rA_prev,:,:rA_k]=G_A
            G_Z[rA_prev:, :,rA_k:]=G_B
        
        coresZ.append(G_Z)
    ranks.append(1)
    S=tt_to_tensor(coresZ)
    return ranks,coresZ,S