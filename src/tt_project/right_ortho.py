import numpy as np

def right_ortho(cores):
    d = len(cores)
    cores_orth = [0 for _ in range(d)]
    cores_orth[-1]=cores[-1] #store the last core

    for k in range(d - 1, 0, -1):
        G_k = cores_orth[k]
        r_prev, n_k, r_k = G_k.shape

        G_k_mat = G_k.reshape(r_prev,-1)
        G_k_mat_T = G_k_mat.T
        Q_t, R_t = np.linalg.qr(G_k_mat_T)

        Q = Q_t.T 
        R = R_t.T  
        r=Q.shape[0]
        cores_orth[k] = Q.reshape(r, n_k, r_k) #store the orthogonalized core

        G_prev = cores[k - 1]
        r_prev2, n_prev, r_p = G_prev.shape

        if r_p != r_prev:
            print("Warning: Inconsistent ranks between cores. Adjusting the previous core's rank.")

        G_prev_mat = G_prev.reshape(-1, r_p)
        G_prev_mat = G_prev_mat @ R # Update the previous core by multiplying with R
        cores_orth[k - 1] = G_prev_mat.reshape(r_prev2, n_prev, r) #store the updated previous core

    return cores_orth