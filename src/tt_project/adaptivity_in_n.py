import numpy as np
from numpy.polynomial import chebyshev as cheb
from tt_project.interval_tree import IntervalTree

def f(x,y):
    return 1/np.abs(x-y)

      
def find_subinterval(domain,n,axis,nbr_of_y_samples,f,node,depth,max_depth,tol=1e-6):
    
    resolved=[]
    other_axis=1-axis
    fixed_samples=np.random.uniform(domain[other_axis,0], domain[other_axis,1], size=nbr_of_y_samples)
    values_test = np.linspace(domain[axis,0], domain[axis,1], 1000,endpoint=True)
    for fixed_sample in fixed_samples:
        p = cheb.Chebyshev.interpolate(lambda x: f(x, fixed_sample),deg=n - 1, domain=[domain[axis,0], domain[axis,1]])
        f_values = f(values_test, fixed_sample)
        p_values = p(values_test)
        max_error = np.max(np.abs(f_values - p_values))
        resolved.append(max_error < tol)

    if np.all(resolved):
        return 
    
    if depth>=max_depth:
        raise ValueError("Maximum number of splits reached. The function may not be resolved within the given tolerance.")
    
    m=(domain[axis,1]+domain[axis,0])/2
    left_domain = domain.copy()
    left_domain[axis] = [domain[axis, 0],m]

    right_domain = domain.copy()
    right_domain[axis] = [m,domain[axis, 1]]
    left_node=node.add_child(IntervalTree(left_domain[axis, 0], left_domain[axis, 1], parent=node))
    right_node=node.add_child(IntervalTree(right_domain[axis, 0], right_domain[axis, 1], parent=node))
    find_subinterval(left_domain,n,axis,nbr_of_y_samples,f,left_node,depth=depth+1,max_depth=max_depth,tol=tol)
    find_subinterval(right_domain,n,axis,nbr_of_y_samples,f,right_node,depth=depth+1,max_depth=max_depth,tol=tol)


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

    # # don't thibk thsi is what we want 
    # for x_leaf in x_root.leaves():
    #     y_root=IntervalTree(domain[axis, 0], domain[axis, 1])
    #     find_subinterval(np.array([[x_leaf.lo,x_leaf.hi]]+[[domain[axis,0],domain[axis,1]]]),n,axis,nbr_of_x_samples,f,y_root,depth=0,max_depth=20,tol=1e-6)
    #     y_root_intervals=y_root.intervals()
    #     print("the resolved intervals in y are :"+str(y_root_intervals))
    #     nbr_intervals_y=len(y_root_intervals)
    #     n1=n1+nbr_intervals_y*n
    #     print("the number of resolved intervals in y is :"+str(nbr_intervals_y))
    #     print("N_2 is equal to "+str(n*(nbr_intervals_y)))
    #     x_leaf.y_tree = y_root

if __name__ == "__main__":
    main() 