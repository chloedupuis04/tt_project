import numpy as np
from tt_project.TTK_adaptivity import *
from mpl_toolkits import mplot3d

def f(x, y):
        return 1/np.abs(x - y)

def f_polynomial(x, y):
    return 1 + 2*x - 3*y + x**2 + x*y + 2*y**2

def f_genz_corner_peak(x, y):
    c1 = 10.0
    c2 = 10.0
    return (1.0 + c1*x + c2*y)**(-3)

functions_smooth= {"Polynomial": f_polynomial, "Genz corner peak function": f_genz_corner_peak}



def main():
   
    print("easy and medium fucniton")

    for name,f in functions_smooth.items():
        fig_error, ax_error = plt.subplots()
        domain= np.array([[0, 1], [0, 1]], dtype=float)
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
                print("Experiment {} with {} samples ({})".format(i2, samples[i1], name))
                tt_cores, list_n, list_intervals = TTK_adaptivity_offline(n,domain,samples[i1],f)
                K = TTK_adaptivity_online(n,domain,tt_cores,list_n,list_intervals,x_points,y_points,d,N_s,N_t,tol )
                K_exact= f(x_points[0, :, None], y_points[0, None, :])
                nError[i1, i2] = (np.linalg.norm(K - K_exact)/ np.linalg.norm(K_exact))

        median_err = np.median(nError, axis=1)
        q25_err = np.percentile(nError, 25, axis=1)
        q75_err = np.percentile(nError, 75, axis=1)

        # Add this value of s to the common first graph
        line, = ax_error.semilogy(samples,median_err,marker="o",label=name)
        ax_error.fill_between(samples,q25_err,q75_err,alpha=0.25,color=line.get_color())
        ax_error.set_xlabel("Number of samples")
        ax_error.set_ylabel("Relative error")
        ax_error.set_title("Error with change in the number of fixed variables")
        ax_error.legend()
        ax_error.grid(True, which="both")

        fig_error.tight_layout()
        fig_error.savefig(f"tests/figures/error_TTK_vs_samples{name}.pdf")
        plt.close(fig_error)

        #second plot 
        N_plot = 2500

        x_grid = np.linspace(domain[0, 0],domain[0, 1],N_plot)

        y_grid = np.linspace(domain[1, 0],domain[1, 1],N_plot)

        x_grid_pts = x_grid[None, :]
        y_grid_pts = y_grid[None, :]

        tt_cores, list_n, list_intervals = TTK_adaptivity_offline(n,domain,nbr_samples,f)

        K_grid = TTK_adaptivity_online(n,domain,tt_cores,list_n,list_intervals,x_grid_pts,y_grid_pts,d, N_plot,N_plot,tol)

        K_exact_grid = f(x_grid_pts[0, :, None], y_grid_pts[0, None, :])

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
        ax_matrix.set_title(fr"Absolute interpolation error for {name}")
        fig_matrix.tight_layout()
        fig_matrix.savefig(f"tests/figures/error_matrix{name}.pdf")
        plt.close(fig_matrix)

        # third plot : plot the interpolation in 3d 
        X, Y = np.meshgrid(x_grid_pts, y_grid_pts)

        fig_3d = plt.figure()
        ax = plt.axes(projection='3d')

        ax.plot_surface(X, Y, K_grid, rstride=10, cstride=10,
                    cmap="magma", edgecolor='none')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x,y)')

        fig_3d.savefig(f"tests/figures/interpolation_3d_{name}.pdf")

        # fourth plot : plot the error in 3d 
            
        X, Y = np.meshgrid(x_grid_pts, y_grid_pts)

        fig_3d = plt.figure()
        ax = plt.axes(projection='3d')

        ax.plot_surface(X, Y, absolute_error, rstride=10, cstride=10,
                    cmap="magma", edgecolor='none')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x,y)')
        fig_3d.savefig(f"tests/figures/error_3d_{name}.pdf")

    
      

    print("hard function")
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
        ax_matrix.set_title(fr"Absolute interpolation error for, $s={s:.0e}$")

        fig_matrix.tight_layout()

        # Different filename for each value of s
        s_name = f"{s:.0e}".replace("-", "m")

        fig_matrix.savefig(f"tests/figures/error_matrix_s_{s_name}.pdf")

        plt.close(fig_matrix)

        # third plot : plot the interpolation in 3d 
        X, Y = np.meshgrid(x_grid_pts, y_grid_pts)

        fig_3d = plt.figure()
        ax = plt.axes(projection='3d')

        ax.plot_surface(X, Y, K_grid, rstride=10, cstride=10,
                    cmap="magma", edgecolor='none')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x,y)')

        fig_3d.savefig(f"tests/figures/interpolation_3d_{s_name}.pdf")

        # fourth plot : plot the error in 3d 
            
        X, Y = np.meshgrid(x_grid_pts, y_grid_pts)

        fig_3d = plt.figure()
        ax = plt.axes(projection='3d')

        ax.plot_surface(X, Y, absolute_error, rstride=10, cstride=10,
                    cmap="magma", edgecolor='none')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x,y)')
        fig_3d.savefig(f"tests/figures/error_3d_{s_name}.pdf")

        # Finish and save the common first graph
    ax_error.set_xlabel("Number of samples")
    ax_error.set_ylabel("Relative error")
    ax_error.set_title("Error with change in the number of fixed variables")
    ax_error.legend()
    ax_error.grid(True, which="both")

    fig_error.tight_layout()
    fig_error.savefig(f"tests/figures/error_TTK_vs_samples.pdf")
    plt.close(fig_error)

if __name__ == "__main__":
    main()

