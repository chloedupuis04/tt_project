
import numpy as np
import time 
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import Matern
from tt_project.PTTK import PTTK_offline,PTTK_online


def matern_kernel(x,y,theta):
    kernel = Matern(length_scale=theta[0], nu=theta[1])
    x = np.asarray(x).reshape(1, -1)
    y = np.asarray(y).reshape(1, -1)

    return kernel(x, y)[0, 0]

def f_matern(x,theta,y):
    return matern_kernel(x,y,theta)

def main():
    print("--------------------------------------")
    print(" ")
    d=3
    d_theta=2
    n=0

    a_s=0
    b_s=1
    Ns=100
    x_points=a_s+(b_s-a_s)*np.random.rand(d,Ns)

    a_t=1
    b_t=2
    Nt=100
    y_points=a_t+(b_t-a_t)*np.random.rand(d,Nt)

    Db=np.sqrt(3)
    a_theta=Db/2
    b_theta=Db
    length_scale=np.random.uniform(a_theta, b_theta,size=1)
    nu=np.random.uniform(1,3,size=1)
    theta=np.array([length_scale[0],nu[0]])

    domain=np.array([[a_s,b_s]]*d+[[a_theta,b_theta]]+[[1,3]]+[[a_t,b_t]]*d)
    print("varying n")
    n=[2,3,4,5,6,8]
    errors=[]
    for n_i in n:
        print("n is ="+str(n_i))
        cores_approx, ranks_approx=PTTK_offline(f_matern,domain,n_i,x_points,y_points,d,d_theta,Ns,Nt,tol=1e-6)
        K_approx=PTTK_online(theta, d_theta,d,domain,n_i,cores_approx,ranks_approx)
        K=[[ f_matern(x_points[:,i],theta,y_points[:,j]) for j in range(Nt)] for i in range(Ns)]
        err=np.linalg.norm(K-K_approx)/np.linalg.norm(K)
        print("the relative error is :"+str(err))
        errors.append(err)

    plt.figure(figsize=(7, 4.5))
    plt.semilogy(n, errors, color='green', marker='o',markersize=7, label="PPTK error")
    plt.grid(True)
    plt.xlabel("Number of Chebyshev nodes n ")
    plt.ylabel("Relative error")
    plt.savefig("figures/error_vs_nbr_chebnodes.pdf")
    plt.close()
    

    print("varying the tolerance")
    n=6
    tols=[1e-4,1e-5,1e-6,1e-8,1e-10]
    errors=[]
    for tol in tols:
        print("n is ="+str(n))
        cores_approx,ranks_approx=PTTK_offline(f_matern,domain,n,x_points,y_points,d,d_theta,Ns,Nt,tol)
        K_approx=PTTK_online(theta, d_theta,d,domain,n,cores_approx,ranks_approx)
        K=[[ f_matern(x_points[:,i],theta,y_points[:,j]) for j in range(Nt)] for i in range(Ns)]
        err=np.linalg.norm(K-K_approx)/np.linalg.norm(K)
        print("the relative error is :"+str(err))
        errors.append(err)

    plt.figure(figsize=(7, 4.5))
    plt.semilogx(tols, errors, color='red', marker='o',markersize=7, label="PPTK error")
    plt.grid(True)
    plt.xlabel(r"Prescribed tolerance $\varepsilon$")
    plt.ylabel("Relative error")
    plt.savefig("figures/error_vs_tol.pdf")
    plt.close()

    print("online time vs N_s and N_t")
    n=6
    list_N=[50,100,250,500]
    Time=[]

    for N_i in list_N:
        Ns_i = N_i
        Nt_i = N_i
        print("N_t is ="+str(Nt_i)+" and N_s is ="+str(Ns_i))
        x_points=a_s+(b_s-a_s)*np.random.rand(d,Ns_i)
        y_points=a_t+(b_t-a_t)*np.random.rand(d,Nt_i)
        cores_approx,ranks_approx=PTTK_offline(f_matern,domain,n,x_points,y_points,d,d_theta,Ns_i,Nt_i,tol=1e-6)
        ta=time.perf_counter()
        K_approx=PTTK_online(theta, d_theta,d,domain,n,cores_approx,ranks_approx)
        tb=time.perf_counter()
        Time.append(tb-ta)
    plt.figure(figsize=(7, 4.5))
    plt.semilogy(list_N, Time, marker='o', color='blue', markersize=7,label="PPTK online time")
    plt.grid(True)
    plt.xlabel("Number of points $N_s$ and $N_t$")
    plt.ylabel("Online time (seconds)")
    plt.savefig("figures/online_times.pdf")
    plt.close()


if __name__ == "__main__":
    main()