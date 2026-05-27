def horizontal_core(core):
    #reshape a core of dimensions (r_prev,n,r) into horizontal unfolding of dimensions (r_prev,n*r)
    r_prev, n, r = core.shape
    return core.reshape(r_prev, n * r)