import numpy as np
from tt_project.tt_svd_delta import tt_svd_delta
from tt_project.right_ortho import right_ortho
from tt_project.left_ortho import left_ortho
from tt_project.delta_rank import delta_rank 
from tt_project.tt_to_tensor import tt_to_tensor


def tt_rounding(cores,eps=1e-6):


    cores_r_orth = right_ortho(cores)
    d=len(cores_r_orth)
    ranks=[]
    ranks.append(1)

    delta=(eps/np.sqrt(d-1))*np.linalg.norm(np.linalg.norm(cores_r_orth[0]))#||A||_F=||G_1||_F as the cores are right-orthogonal   

    for k in range(d-1):
        G_k=cores_r_orth[k]
        r_prev,n_k,r_k=G_k.shape
        G_k_mat=G_k.reshape(r_prev*n_k,r_k)
        Q,R=np.linalg.qr(G_k_mat)

        U,S,Vt=np.linalg.svd(R,full_matrices=False)
    
        r_delta=delta_rank(S,delta)
        ranks.append(r_delta)
        U_k= U[:,:r_delta]
        S_k= S[:r_delta]
        Vt_k= Vt[:r_delta,:]

        G_k_mat=Q@U_k
        cores_r_orth[k]=G_k_mat.reshape(r_prev,n_k,r_delta)

        G_next=cores_r_orth[k+1]
        rk,n_k_next,r_k_next=G_next.shape
        G_next_mat=G_next.reshape(rk,-1)
        G_next_mat=np.diag(S_k)@Vt_k@G_next_mat
        cores_r_orth[k+1]=G_next_mat.reshape(r_delta,n_k_next,r_k_next)
    
    ranks.append(1)
    return cores_r_orth,ranks