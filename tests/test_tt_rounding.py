    
import numpy as np
import matplotlib.pyplot as plt
from tt_project.sum_tensor import sum_tensor
from tt_project.tt_rounding import tt_rounding
from tt_project.tt_to_tensor import tt_to_tensor


def main ():
    coresX=[np.random.rand(1,10,4),np.random.rand(4,10,8),np.random.rand(8,10,3),np.random.rand(3,10,1)]
    coresY=[np.random.rand(1,10,3),np.random.rand(3,10,6),np.random.rand(6,10,4),np.random.rand(4,10,1)]
    eta=1e-4
    coresY[0]=eta*coresY[0]
    ranks_before,cores_sum,S=sum_tensor(coresX,coresY)
    
    # Different rounding tolerances
    epsilons = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
    errors = []
    all_ranks_after = []

    for eps in epsilons:
        cores_after, ranks_after = tt_rounding(cores_sum, eps)

        # Compute full tensor from rounded TT
        S_round = tt_to_tensor(cores_after)   

        rel_error = np.linalg.norm(S - S_round) / np.linalg.norm(S)

        errors.append(rel_error)
        all_ranks_after.append(ranks_after)
        
    
    plt.figure(figsize=(7, 4.5))
    plt.loglog(epsilons, errors, marker='o', markersize=7,color ="red",label="TT-rounding error")
    plt.loglog(epsilons, epsilons, linestyle='--',color ="grey",label=r"$Error=\varepsilon$")

    plt.xlabel(r"Prescribed tolerance $\varepsilon$")
    plt.ylabel("Relative error")
    plt.grid(True)
    plt.legend() 
    plt.savefig("figures/TT-rounding_error.pdf")
    plt.close()

    plt.figure(figsize=(7, 4.5))
    k=np.arange(len(ranks_before))
    for eps, ranks in zip(epsilons, all_ranks_after):
        plt.plot(k, ranks, marker='o', markersize=7,label=fr"$\varepsilon={eps:.0e}$")
    plt.plot(k, ranks_before, marker='x',linestyle='--', color ="grey",label="Before rounding")

    plt.xlabel(r"Rank index $k$")
    plt.xticks(k)
    plt.ylabel(r"TT-rank $r_k$")
    plt.grid(True)
    plt.legend()
    plt.savefig("figures/ranks_variation.pdf")
    plt.close()


if __name__ == "__main__":
    main()