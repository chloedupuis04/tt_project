# Testing the greedy2_cross function (Python's version)

import numpy as np
from tt_project.greedy2_cross import *
from tt_project.adaptivity_in_n import find_subinterval
from numpy.polynomial import chebyshev as cheb
from tt_project.interval_tree import IntervalTree

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
# We first define the function that defines the entries of our tensor


n=8
nbr_of_y_samples=10
domain=np.array([[0,1]]+[[1,2]], dtype=float)
axis=0
x_root = IntervalTree(domain[axis, 0], domain[axis, 1])
find_subinterval(domain,n,axis,nbr_of_y_samples,f,x_root,depth=0,max_depth=100,tol=1e-6)
x_intervals=x_root.intervals()
n0=n*len(x_intervals)


nbr_of_x_samples=10
axis=1
y_root=IntervalTree(domain[axis, 0], domain[axis, 1])
find_subinterval(domain,n,axis,nbr_of_x_samples,f,y_root,depth=0,max_depth=20,tol=1e-6)
y_intervals=y_root.intervals()
n1=n*len(y_intervals)



ns = [n0, n1,] # List of dimensions
tol = 1e-4 # Tolerance for TT Cross Greedy

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
M_approx = np.zeros((n0, n1))
for i0 in range(n0):
    for i1 in range(n1):
            M_true[i0,  i1] = func_silly_aux([i0, i1])

# Obtain the full tensor from the TT by hand (not optimal)
M_approx = np.reshape(tt_cores[1], (2*10, 2)) @ np.reshape(tt_cores[2], (2, 10))
M_approx = np.reshape(tt_cores[0], (10, 2)) @ np.reshape(M_approx, (2, 100))
M_approx = np.reshape(M_approx, (10, 10, 10))

err = M_true - M_approx
err = np.linalg.norm(err.ravel())
normM = np.linalg.norm(M_true.ravel())
print("Relative Frobenius error between the true tensor and the TT approximation: ", err/normM)