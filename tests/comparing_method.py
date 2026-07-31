import numpy as np
from tt_project.adaptive_TT import *
import pandas as pd



def f_genz_oscillatory(x, y):
    parameters = np.loadtxt(
    "/Users/coco/Desktop/tt_project/tests/genz_oscillatory_parameters.csv",
    delimiter=",",
    skiprows=1)
    w, c1, c2 = parameters
    return np.cos( 2 * np.pi * w + c1 * (x + 1) / 2+ c2 * (y + 1) / 2)


def f_genz_corner_peak(x, y):
    parameters = np.loadtxt(
    "/Users/coco/Desktop/tt_project/tests/genz_corner_peak_parameters.csv",
    delimiter=",",
    skiprows=1)
    c1,c2=parameters
    return (1.0 + c1*(x+1)/2 + c2*(y+1)/2)**(-3)

def f_ackley(x,y):
    return -20*np.exp(-0.2*np.sqrt(1/7 *(x**2+y**2)))-np.exp(1/7*(np.cos(2*np.pi*x)+np.cos(2*np.pi*x))) +20+np.exp(1)


functions_smooth= {"ackley":f_ackley,"genz_corner_peak": f_genz_corner_peak,"genz_oscillatory": f_genz_oscillatory}

def main():
    for name,f in functions_smooth.items():
        #ploting test 
        n = 8
        nbr_samples =100
        d = 1
        N_grid=100
        if name=="ackley":
            domain=np.array([[-32.768, 32.768], [-32.768, 32.768]], dtype=float)
        else:
            domain = np.array([[-1, 1], [-1, 1]], dtype=float)
        tol=1e-10
        tt_cores, list_n, list_intervals = adaptive_tt_offline(n, domain, nbr_samples, f, tol, 2000)

        x_grid = np.linspace(domain[0, 0], domain[0, 1], N_grid)
        y_grid = np.linspace(domain[1, 0], domain[1, 1], N_grid)
        X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")

        x_points = x_grid.reshape(1, -1)
        y_points = y_grid.reshape(1, -1)

        K = adaptive_tt_online(n, domain, tt_cores, list_n, list_intervals,x_points, y_points, d, N_grid, N_grid)

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, K, cmap="viridis", edgecolor="none")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("interpolant value")
        ax.set_title(f"Interpolant: {name}")
        plt.savefig(f"tests/figures/interp_3d_{name}.pdf")
        plt.close()

        F_true = f(X, Y)

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y,F_true,cmap="viridis",edgecolor="none")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("function value")
        ax.set_title(f"True function: {name}")
        plt.savefig(f"tests/figures/function_3d_{name}.pdf")
        plt.close()

        abs_error = np.abs(F_true - K)

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, abs_error, cmap="viridis", edgecolor="none")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("absolute error")
        ax.set_title(f"Absolute error: {name}")
        plt.savefig(f"tests/figures/error_3d_{name}.pdf")
        plt.close()
        

        # test n vs tol
        print("test1")
        n = 8
        nbr_samples =100
        d = 1
        N_s = 1000
        N_t = 1000
        tol_list=[1e-6,1e-8,1e-10,1e-12,1e-14]
        domain = np.array([[-1, 1], [-1, 1]], dtype=float)
        print(domain)
        x_points = domain[0, 0] + (domain[0, 1] - domain[0, 0]) * np.random.rand(d, N_s)
        y_points = domain[1, 0] + (domain[1, 1] - domain[1, 0]) * np.random.rand(d, N_t)

        nExp = 4
        nError = np.zeros((len(tol_list), nExp), dtype=float)
        nCheb_x = np.zeros((len(tol_list), nExp), dtype=float)
        nCheb_y= np.zeros((len(tol_list), nExp), dtype=float)

        for i1 in range(len(tol_list)):
            for i2 in range(nExp):
                tt_cores, list_n, list_intervals = adaptive_tt_offline( n, domain, nbr_samples, f, tol_list[i1],100)
                nCheb_x[i1, i2] = list_n[0]
                nCheb_y[i1, i2] = list_n[1]
  
        eftt = np.genfromtxt(f"tests/EFTT_results_{name}.csv",delimiter=",",names=True)

        tol_EFTT = eftt["tol"]
        nCheb_x_EFTT = eftt["nChebX"].astype(int)
        nCheb_y_EFTT = eftt["nChebY"].astype(int)
        err_EFTT = eftt["relativeError"]
        plt.figure()
        mean_nCheb_x = np.mean(nCheb_x, axis=1)
        plt.semilogx(tol_list,mean_nCheb_x, "-o",label="our method")
        plt.semilogx(tol_list,nCheb_x_EFTT, "-x",label="EFTT")
        plt.xlabel('tolerance')
        plt.ylabel('nbr Cheb nodes')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"tests/figures/nx_vs_tol for {name}.pdf")
        plt.close()

        plt.figure()
        mean_nCheb_y = np.mean(nCheb_y, axis=1)
        plt.semilogx(tol_list,mean_nCheb_y, "-o",label="our method")
        plt.semilogx(tol_list,nCheb_y_EFTT, "-x",label="EFTT")
        plt.xlabel('tolerance')
        plt.ylabel('nbr Cheb nodes')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"tests/figures/ny_vs_tol for {name}.pdf")
        plt.close()
    
        #test err vs n
        print("test err vs n")
        nExp = 5
        tol=1e-9
        max_nbr_intervals_list=[10,20,25,30,35]
        nError = np.zeros((len(max_nbr_intervals_list), nExp), dtype=float)
        nCheb_x = np.zeros((len(max_nbr_intervals_list), nExp), dtype=float)
        nCheb_y= np.zeros((len(max_nbr_intervals_list), nExp), dtype=float)

        for i1 in range(len(max_nbr_intervals_list)):
            for i2 in range(nExp):
                tt_cores, list_n, list_intervals = adaptive_tt_offline( n, domain, nbr_samples, f, tol,max_nbr_intervals_list[i1])
                nCheb_x[i1, i2] = list_n[0]
                nCheb_y[i1, i2] = list_n[1]
                K = adaptive_tt_online( n, domain, tt_cores, list_n, list_intervals,x_points, y_points, d, N_s, N_t)
                K_exact = f(x_points[0, :, None], y_points[0, None, :])
                nError[i1, i2] = np.linalg.norm(K - K_exact) / np.linalg.norm(K_exact)
        plt.figure()
        mean_err = np.mean(nError, axis=1)
        mean_nX = np.mean(nCheb_x, axis=1)

        plt.semilogy(mean_nX,mean_err, "-o",label="our method")
        plt.semilogy(nCheb_x_EFTT,err_EFTT, "-x",label="EFTT")
        plt.xlim( min(min(mean_nX), min(nCheb_x_EFTT)),
        max(max(mean_nX), max(nCheb_x_EFTT))
        )
        plt.xlabel('nbr cheb nodes in X')
        plt.ylabel('Relative error (mean over experiments)')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"tests/figures/error_TTK_vs_nX{name}.pdf")
        plt.close()

        plt.figure()
        mean_err = np.mean(nError, axis=1)
        mean_nY = np.mean(nCheb_y, axis=1)
        plt.semilogy(mean_nY,mean_err, "-o",label="our method")
        plt.semilogy(nCheb_y_EFTT,err_EFTT, "-x",label="EFTT")
        plt.xlim( min(min(mean_nY), min(nCheb_y_EFTT)),
        max(max(mean_nY), max(nCheb_y_EFTT))
        )
        plt.xlabel('nbr cheb nodes in Y')
        plt.ylabel('Relative error (mean over experiments)')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"tests/figures/error_TTK_vs_nY{name}.pdf")
        plt.close()

        #test matrix color plot
        max_nbr_intervals_list=[10,15,20,25,30]
        N_plot = 2500

        x_grid = np.linspace(domain[0, 0],domain[0, 1],N_plot)

        y_grid = np.linspace(domain[1, 0],domain[1, 1],N_plot)

        x_grid_pts = x_grid[None, :]
        y_grid_pts = y_grid[None, :]
        for i1 in range(len(max_nbr_intervals_list)):
            print("with nbr max"+str(max_nbr_intervals_list[i1]))
            tt_cores, list_n, list_intervals = adaptive_tt_offline(n,domain,nbr_samples,f,tol,max_nbr_intervals_list[i1])

            K_grid = adaptive_tt_online(n,domain,tt_cores,list_n,list_intervals,x_grid_pts,y_grid_pts,d, N_plot,N_plot)

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
            ax_matrix.set_title(fr"Absolute interpolation error for{name} with {max_nbr_intervals_list[i1]}.pdf")
            fig_matrix.tight_layout()
            fig_matrix.savefig(f"tests/figures/error_matrix{name} with {max_nbr_intervals_list[i1]}.pdf")
            plt.close(fig_matrix)
    

if __name__ == "__main__":
    main()