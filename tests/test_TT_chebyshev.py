
import numpy as np
import matplotlib.pyplot as plt
from tt_project.TT_chebyshev import TT_Chebyshev_interpolation, TT_Chebyshev_evaluation

def f(theta):
    theta=np.asarray(theta)
    return np.exp(-np.sum(theta**2))
def f2(theta):
    theta=np.asarray(theta)
    return np.exp(np.sum(np.abs(theta)))

def corner_peak(theta):
    theta = np.asarray(theta)
    d = len(theta)
    c = np.ones(d)
    return (1 + np.sum(c * theta)) ** (-(d + 1))

def plots_various_n(d_theta,nbr_theta,domain,f,n_values,name_plots1,name_plots2):
    errors=[]
    total_entries=[]
    for n in n_values:
        print("n is ="+str(n))
        cores_afterrounding, ranks_rounding=TT_Chebyshev_interpolation(f,domain,n, d_theta, 1e-6)

        shape=tuple([nbr_theta]*d_theta)
        K_approx=np.zeros(shape)
        K=np.zeros(shape)
        thetas_values=[np.random.uniform(domain[j,0],domain[j,1],size=nbr_theta) for j in range(d_theta)]
        for idx in np.ndindex(*K.shape): 
                theta_point=np.array([thetas_values[j][idx[j]] for j in range(d_theta)])
                K_approx[idx]=TT_Chebyshev_evaluation(theta_point,d_theta,domain,n,cores_afterrounding,ranks_rounding)[0,0]
                K[idx]=f(theta_point)
        error = np.linalg.norm(K-K_approx)/np.linalg.norm(K)
        errors.append(error)
        total_entries.append(cores_afterrounding.total_entries())

    plt.semilogy(n_values, errors, color='green', marker='o',markersize=7, label="TT-Chebyshev error")
    plt.xlabel("Number of Chebyshev nodes n ")
    plt.ylabel("Relative error")
    plt.grid(True)
    plt.savefig(name_plots1)
    plt.close()

    plt.semilogy(n_values, total_entries, color='blue', marker='s',markersize=7, label="Total entries in TT representation")
    plt.xlabel("Number of Chebyshev nodes n ")
    plt.ylabel("Total entries in TT representation")
    plt.grid(True)
    plt.savefig(name_plots2)
    plt.close()

def main():

   

    print("test 2")
    n_values=[2,4,6,8,10,12]
    d_theta=3
    nbr_theta=8
    domain=np.array([[-1,1]]*d_theta)
    plots_various_n(d_theta,nbr_theta,domain,f,n_values,"tests/figures/error_vs_nbr_chebnodes_f1.pdf","tests/figures/total_entries_vs_nbr_chebnodes_f1.pdf")
    plots_various_n(d_theta,nbr_theta,domain,f2,n_values,"tests/figures/error_vs_nbr_chebnodes_f2.pdf","tests/figures/total_entries_vs_nbr_chebnodes_f2.pdf")

    domain=np.array([[0,1]]*d_theta)
    plots_various_n(d_theta,nbr_theta,domain,corner_peak,n_values,"tests/figures/error_vs_nbr_chebnodes_corner_peak.pdf","tests/figures/total_entries_vs_nbr_chebnodes_corner_peak.pdf")
  



if __name__ == "__main__":
    main()  