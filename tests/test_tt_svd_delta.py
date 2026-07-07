
import numpy as np
import matplotlib.pyplot as plt
from tt_project.tt_svd import tt_svd
from tt_project.tt_to_tensor import tt_to_tensor
from tt_project.hilbert_tensor import hilbert_tensor
from tt_project.delta_rank import delta_rank
from tt_project.tt_svd_delta import tt_svd_delta
from tt_project.square_root_sum_tensor import sqrt_tensor


def main():

    A = hilbert_tensor([5,8,13,11,16,18])
    B= sqrt_tensor([5,8,13,11,16,18], 1, 2)

    errors_A = []
    errors_B = []
    epsilons = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8,1e-9,1e-10,1e-11,1e-12,1e-13]

    for eps in epsilons:

        cores_A,_,_= tt_svd_delta(A, eps)
        A_approx = tt_to_tensor(cores_A)
        err_A = np.linalg.norm(A - A_approx) / np.linalg.norm(A)
        errors_A.append(err_A)

        cores_B,_,_= tt_svd_delta(B, eps)
        B_approx = tt_to_tensor(cores_B)
        err_B = np.linalg.norm(B - B_approx) / np.linalg.norm(B)
        errors_B.append(err_B)


    plt.figure(figsize=(7, 4.5))
    plt.loglog(epsilons, errors_A, color='red', marker='o',markersize=7, label="Hilbert tensor")
    plt.loglog(epsilons, errors_B, color='blue', marker='s', markersize=7,label="Square root sum tensor")
    plt.loglog(epsilons, epsilons, linestyle='--', color='grey',  label="Reference $Error=\\varepsilon$")
    plt.xlabel("Prescribed tolerance $\\varepsilon$")
    plt.ylabel("Relative error")
    plt.title("TT-SVD approximation error")
    plt.grid(True)
    plt.legend()
    plt.savefig("tests/figures/TT-SVD_error.pdf")
    plt.close()



if __name__ == "__main__":
    main()