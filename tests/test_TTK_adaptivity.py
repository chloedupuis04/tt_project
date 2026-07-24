import numpy as np
from tt_project.TTK_adaptivity import *

def f(x, y):
        return 1/np.abs(x - y)

def main():
   
    print("test 1")

    S = [1e-2, 1e-3, 1e-4, 1e-6]

    fig_error, ax_error = plt.subplots()

    for s in S:

        domain = np.array([[0, 1 - s / 2],[1 + s / 2, 2]],dtype=float)

        n = 8
        nbr_samples = 100
        d = 1
        N_s = 1000
        N_t = 1000
        tol = 1e-6

        x_points = (domain[0, 0]+(domain[0, 1]-domain[0, 0])*np.random.rand(d, N_s))

        y_points = (domain[1, 0]+(domain[1, 1]-domain[1, 0])* np.random.rand(d, N_t))

        nExp = 8
        samples = [50, 100, 200, 300, 400, 500]
        nSamples = len(samples)

        nError = np.zeros((nSamples, nExp), dtype=float)

        for i1 in range(nSamples):
            for i2 in range(nExp):
                print("Experiment {} with {} samples and separation {}".format(i2, samples[i1],s))
                tt_cores, list_n, list_intervals = TTK_adaptivity_offline(n,domain,samples[i1],f)
                K = TTK_adaptivity_online(n,domain,tt_cores,list_n,list_intervals,x_points,y_points,d,N_s,N_t,tol )
                K_exact = 1 / np.linalg.norm(x_points[:, :, None] - y_points[:, None, :],axis=0)
                nError[i1, i2] = (np.linalg.norm(K - K_exact)/ np.linalg.norm(K_exact))

        median_err = np.median(nError, axis=1)
        q25_err = np.percentile(nError, 25, axis=1)
        q75_err = np.percentile(nError, 75, axis=1)

        # Add this value of s to the common first graph
        line, = ax_error.semilogy(samples,median_err,marker="o",label=fr"$s={s:.0e}$")

        ax_error.fill_between(samples,q25_err,q75_err,alpha=0.25,color=line.get_color())

        #second plot 
        N_plot = 2500

        x_grid = np.linspace(domain[0, 0],domain[0, 1],N_plot)

        y_grid = np.linspace(domain[1, 0],domain[1, 1],N_plot)

        x_grid_pts = x_grid[None, :]
        y_grid_pts = y_grid[None, :]

        tt_cores, list_n, list_intervals = TTK_adaptivity_offline(n,domain,nbr_samples,f)

        K_grid = TTK_adaptivity_online(n,domain,tt_cores,list_n,list_intervals,x_grid_pts,y_grid_pts,d, N_plot,N_plot,tol)

        K_exact_grid = 1 / np.linalg.norm(x_grid_pts[:, :, None] - y_grid_pts[:, None, :],axis=0)

        absolute_error = np.abs(K_exact_grid - K_grid)

        fig_matrix, ax_matrix = plt.subplots()

        image = ax_matrix.imshow(absolute_error.T,extent=[domain[0, 0],domain[0, 1],domain[1, 0],domain[1, 1]],
            origin="lower",
            aspect="auto",
            cmap="magma",
            norm=LogNorm(vmin=max(absolute_error.min(), 1e-12),vmax=absolute_error.max())
        )

        fig_matrix.colorbar(image,ax=ax_matrix,label=r"$|f(x,y)-p(x,y)|$")

        ax_matrix.set_xlabel(r"$x$")
        ax_matrix.set_ylabel(r"$y$")
        ax_matrix.set_title(fr"Absolute interpolation error, $s={s:.0e}$")

        fig_matrix.tight_layout()

        # Different filename for each value of s
        s_name = f"{s:.0e}".replace("-", "m")

        fig_matrix.savefig(f"tests/figures/error_matrix_s_{s_name}.pdf")

        plt.close(fig_matrix)

    # Finish and save the common first graph
    ax_error.set_xlabel("Number of samples")
    ax_error.set_ylabel("Relative error")
    ax_error.set_title("Error with change in the number of fixed variables")
    ax_error.legend()
    ax_error.grid(True, which="both")

    fig_error.tight_layout()
    fig_error.savefig("tests/figures/error_TTK_vs_samples.pdf")
    plt.close(fig_error)

if __name__ == "__main__":
    main()

