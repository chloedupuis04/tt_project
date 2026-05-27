import numpy as np
def tt_to_tensor(cores):

    X = cores[0]
    dims = [cores[0].shape[1]]

    for i in range(1, len(cores)):
        Y = cores[i]
        r_i, n_i, r_next = Y.shape

        X = X.reshape(-1, r_i)
        Y = Y.reshape(r_i, -1)
        X = X @ Y
        X = X.reshape(-1, r_next)
        dims.append(n_i)

    return X.reshape(*dims)
    