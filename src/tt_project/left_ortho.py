import numpy as np

def left_ortho(cores):
    d=len(cores)

    cores_orth=[0 for _ in range (d)]
    cores_orth[0]=cores[0]

    for k in range (d-1):
        G_k=cores_orth[k]
        r_prev,n_k,r_k=G_k.shape

        G_k_mat=G_k.reshape(r_prev*n_k,r_k)

        Q,R=np.linalg.qr(G_k_mat)
        r_new=Q.shape[1]
        cores_orth[k]=Q.reshape(r_prev,n_k,r_new)

        G_next=cores[k+1]
        rk,n_k_next,r_k_next=G_next.shape

        if rk!=r_k:
            print("Warning: Inconsistent ranks between cores.")

        G_next_mat=G_next.reshape(rk,-1)
        G_next_mat=R@G_next_mat
        cores_orth[k+1]=G_next_mat.reshape(r_new,n_k_next,r_k_next)
    
    return cores_orth 