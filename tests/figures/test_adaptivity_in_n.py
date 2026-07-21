from tt_project.adaptivity_in_n import *

def f(x,y):
    return 1/np.abs(x-y)

def main():
    n=8
    nbr_of_y_samples=10
    domain=np.array([[0,1]]+[[1,2]], dtype=float)
    axis=0
    x_root = IntervalTree(domain[axis, 0], domain[axis, 1])
    find_subinterval(domain,n,axis,nbr_of_y_samples,f,x_root,depth=0,max_depth=100,tol=1e-6)
    x_intervals=x_root.intervals()
    print("the resolved intervals are :"+str(x_intervals))
    l0=len(x_intervals)
    print("the number of resolved intervals is :"+str(l0))
    n0=n*l0
    print("N_1 is equal to "+str(n0))

    print("test fuction find_Nk in y")
    nbr_of_x_samples=10
    axis=1
    y_root=IntervalTree(domain[axis, 0], domain[axis, 1])
    find_subinterval(domain,n,axis,nbr_of_x_samples,f,y_root,depth=0,max_depth=20,tol=1e-6)
    y_intervals=y_root.intervals()
    print("the resolved intervals in y are :"+str(y_intervals))
    l1=len(y_intervals)
    print("the number of resolved intervals in y is :"+str(l1))
    n1=n*l1
    print("N_2 is equal to "+str(n1))

    print("test fuction find_Nk in y")
    nbr_of_x_samples=10
    axis=1
    y_root=IntervalTree(domain[axis, 0], domain[axis, 1])
    find_subinterval(domain,n,axis,nbr_of_x_samples,f,y_root,depth=0,max_depth=20,tol=1e-6)
    y_intervals=y_root.intervals()
    print("the resolved intervals in y are :"+str(y_intervals))
    l1=len(y_intervals)
    print("the number of resolved intervals in y is :"+str(l1))
    n1=n*l1
    print("N_2 is equal to "+str(n1))

if __name__ == "__main__":
    main() 