# Testing the greedy2_cross function (Python's version)

import numpy as np
from tt_project.greedy2_cross import *
from tt_project.adaptivity_in_n import *
from numpy.polynomial import chebyshev as cheb
from tt_project.interval_tree import IntervalTree
from tt_project.tt_to_tensor import tt_to_tensor

def phi(alpha, beta, x):
    return (beta - alpha) / 2 * x + (alpha + beta) / 2

def concatenated_nodes(intervals, n):
    nodes = []
    for a, b in intervals:
        nodes.append(phi(a, b, cheb.chebpts1(n)))

    return np.concatenate(nodes)

# Define the function to be interpolated
def f(x, y):
    return 1/np.abs(x - y)


# The following auxiliary function uses the function we want to interpolate
# but the input is the index of the discretization points instead of the actual
# x1, x2, ...

def func_silly_aux(ind):
    return f(x_0_disc[ind[0]], x_1_disc[ind[1]])

def f2(x,y,z):
     return 1/((x-y)**2+(y-z)**2)

def func2_silly_aux(ind):
    return f2(x_disc[ind[0]],y_disc[ind[1]],z_disc[ind[2]])

print("test of a 2D function")

n=8
nbr_of_y_samples=20
domain=np.array([[0,1]]+[[1,2]], dtype=float)
axis=0
x_root = IntervalTree(domain[axis, 0], domain[axis, 1])
find_subinterval_d(domain,n,axis,nbr_of_y_samples,f,x_root,depth=0,max_depth=100,tol=1e-6)
x_intervals=x_root.intervals()
n0=n*len(x_intervals)

nbr_of_x_samples=20
axis=1
y_root=IntervalTree(domain[axis, 0], domain[axis, 1])
find_subinterval_d(domain,n,axis,nbr_of_x_samples,f,y_root,depth=0,max_depth=20,tol=1e-6)
y_intervals=y_root.intervals()
n1=n*len(y_intervals)

ns = [n0, n1,] # List of dimensions
tol = 1e-6 # Tolerance for TT Cross Greedy

# In the interpolation the following should be the chebyshev nodes in the different
# subintervals concatenated together.
x_0_disc = concatenated_nodes(x_intervals, n)
x_1_disc = concatenated_nodes(y_intervals, n)


# This is how to use the function greedy2_cross. The outputs are the following
#   tt_cores : the TT cores
#   Jyl : left global pivot indices in the cross approximation algorithm (don't need this yet)
#   Jyr : right global pivot indices in the cross approximation algorithm (don't need this yet)
#   ilocl : left local pivot indices in the cross approximation algorithm (don't need this yet)
#   ilocr : right local pivot indices in the cross approximation algorithm (don't need this yet)
#   evalcnt : number of function evaluations used in the cross approximation algorithm (don't need this yet)

tt_cores, Jyl, Jyr, ilocl, ilocr, evalcnt = greedy2_cross(ns, func_silly_aux, tol = tol)

print("Shape of tt cores: ", [tt_core.shape for tt_core in tt_cores])

# Test this method by hand
M_true = np.zeros((n0, n1))
for i0 in range(n0):
    for i1 in range(n1):
            M_true[i0,  i1] = func_silly_aux([i0, i1])

# Obtain the full tensor from the TT by hand (not optimal)

M_approx = tt_to_tensor(tt_cores)
err = M_true - M_approx
err = np.linalg.norm(err.ravel())
normM = np.linalg.norm(M_true.ravel())
print("Relative Frobenius error between the true tensor and the TT approximation: ", err/normM)

print("test 2 with 3D function")

n=8
nbr_of_yz_samples=20
domain=np.array([[0,1]]+[[1,2]] +[[1,2]], dtype=float)
axis=0
x_root = IntervalTree(domain[axis, 0], domain[axis, 1])
find_subinterval_d(domain,n,axis,nbr_of_yz_samples,f2,x_root,depth=0,max_depth=100,tol=1e-6)
x_intervals=x_root.intervals()
n0=n*len(x_intervals)

nbr_of_xz_samples=20
axis=1
y_root=IntervalTree(domain[axis, 0], domain[axis, 1])
find_subinterval_d(domain,n,axis,nbr_of_xz_samples,f2,y_root,depth=0,max_depth=100,tol=1e-6)
y_intervals=y_root.intervals()
n1=n*len(y_intervals)

nbr_of_xy_samples=20
axis=2
z_root=IntervalTree(domain[axis, 0], domain[axis, 1])
find_subinterval_d(domain,n,axis,nbr_of_xy_samples,f2,z_root,depth=0,max_depth=100,tol=1e-6)
z_intervals=z_root.intervals()
n2=n*len(z_intervals)

ns = [n0,n1,n2] # List of dimensions
tol = 1e-6# Tolerance for TT Cross Greedy

# In the interpolation the following should be the chebyshev nodes in the different
# subintervals concatenated together.
x_disc = concatenated_nodes(x_intervals, n)
y_disc = concatenated_nodes(y_intervals, n)
z_disc=concatenated_nodes(z_intervals, n)


tt_cores, Jyl, Jyr, ilocl, ilocr, evalcnt = greedy2_cross(ns, func2_silly_aux, tol = tol)

print("Shape of tt cores: ", [tt_core.shape for tt_core in tt_cores])

# Test this method by hand
M_true = np.zeros((n0, n1,n2))
for i0 in range(n0):
    for i1 in range(n1):
            for i2 in range(n2):
                M_true[i0,i1,i2] = func2_silly_aux([i0,i1,i2])

# Obtain the full tensor from the TT by hand (not optimal)
M_approx = tt_to_tensor(tt_cores)
err = M_true - M_approx
err = np.linalg.norm(err.ravel())
normM = np.linalg.norm(M_true.ravel())
print("Relative Frobenius error between the true tensor and the TT approximation: ", err/normM)
