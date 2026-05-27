def v2h(V_z,N_k):
    #function that from a vertical unfoldings compute the horizontal unfloding 
    r_prev_n, r_k = V_z.shape
    n = N_k
    r_prev = r_prev_n // n
    return V_z.reshape(r_prev, n * r_k)