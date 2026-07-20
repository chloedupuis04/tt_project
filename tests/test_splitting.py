# Here we want to test how consistent our partitions are

# Testing the greedy2_cross function (Python's version)

import numpy as np
from tt_project.greedy2_cross import *
from tt_project.adaptivity_in_n import find_subinterval
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

# We test for different number of y/x samples
nExp = 100 # Number of times we run each experiment
samples = [10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100] # Number of samples to test
nSamples = len(samples)
nSplitsX = np.zeros((nSamples, nExp), dtype=int) # Number of splits in x for each experiment
nSplitsY = np.zeros((nSamples, nExp), dtype=int) # Number of splits in x for each experiment
n = 8 # Degree of local Chebyshev coefficient

for i1 in range(nSamples):
    for i2 in range(nExp):
        nbr_of_y_samples = samples[i1]
        print("Experiment {} with {} samples".format(i2, nbr_of_y_samples))
        # Start splitting for x
        domain=np.array([[0,1]]+[[1,2]], dtype=float)
        axis=0
        x_root = IntervalTree(domain[axis, 0], domain[axis, 1])
        find_subinterval(domain,n,axis,nbr_of_y_samples,f,x_root,depth=0,max_depth=100,tol=1e-6)
        x_intervals=x_root.intervals()
        nSplitsX[i1, i2] = len(x_intervals)
        # Splitting for y
        nbr_of_x_samples = samples[i1]
        axis=1
        y_root=IntervalTree(domain[axis, 0], domain[axis, 1])
        find_subinterval(domain,n,axis,nbr_of_x_samples,f,y_root,depth=0,max_depth=20,tol=1e-6)
        y_intervals=y_root.intervals()
        nSplitsY[i1, i2] = len(y_intervals)
        print("Number of splits in x: {}, Number of splits in y: {}".format(nSplitsX[i1, i2], nSplitsY[i1, i2]))

# Plot
