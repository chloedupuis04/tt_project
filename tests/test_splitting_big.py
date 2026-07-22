# Here we want to test how consistent our partitions are

import os
import sys
PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(
    PROJECT_PATH,"src/"
)

sys.path.append(PROJECT_PATH)
sys.path.append(SOURCE_PATH)

# Testing the greedy2_cross function (Python's version)

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend: renders to file, avoids macOS GUI backend crash
import matplotlib.pyplot as plt
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
samples = [10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 400, 500, 750, 1000] # Number of samples to test
nSamples = len(samples)
seps = [1e-1, 1e-4, 1e-8, 1e-12, 1e-16]# Separation between the boxes
colsSeps = ["#8FBBFA", "#63A1F8", "#0B6CF4", "#07459C", "#031E44"]
nSeps = len(seps)
nSplitsX = np.zeros((nSeps, nSamples, nExp), dtype=int) # Number of splits in x for each experiment
nSplitsY = np.zeros((nSeps, nSamples, nExp), dtype=int) # Number of splits in x for each experiment
n = 8 # Degree of local Chebyshev coefficient

for i0 in range(nSeps):
    sep = seps[i0]
    for i1 in range(nSamples):
        nbr_of_y_samples = samples[i1]
        print("Experiment with {} samples".format(nbr_of_y_samples))
        for i2 in range(nExp):
            # Start splitting for x
            domain=np.array([[0,1 - sep/2]]+[[1 + sep/2,2]], dtype=float)
            axis=0
            x_root = IntervalTree(domain[axis, 0], domain[axis, 1])
            find_subinterval(domain,n,axis,nbr_of_y_samples,f,x_root,depth=0,max_depth=100,tol=1e-6)
            x_intervals=x_root.intervals()
            nSplitsX[i0, i1, i2] = len(x_intervals)
            # Splitting for y
            nbr_of_x_samples = samples[i1]
            axis=1
            y_root=IntervalTree(domain[axis, 0], domain[axis, 1])
            find_subinterval(domain,n,axis,nbr_of_x_samples,f,y_root,depth=0,max_depth=20,tol=1e-6)
            y_intervals=y_root.intervals()
            nSplitsY[i0, i1, i2] = len(y_intervals)

# Save to disc
np.savetxt(PROJECT_PATH + "/tests/data/nSplitsX.txt", np.reshape(nSplitsX, nSeps*nSamples*nExp ), fmt = '%d')
np.savetxt(PROJECT_PATH + "/tests/data/nSplitsY.txt", np.reshape(nSplitsY, nSeps*nSamples*nExp ), fmt = '%d')


# Plot: median number of splits vs samples, with 25th-75th percentile error bands.
# Median and percentiles are taken across the nExp experiments (axis=1).
medianX = np.median(nSplitsX, axis=2)
p25X = np.percentile(nSplitsX, 25, axis=2)
p75X = np.percentile(nSplitsX, 75, axis=2)

medianY = np.median(nSplitsY, axis=2)
p25Y = np.percentile(nSplitsY, 25, axis=2)
p75Y = np.percentile(nSplitsY, 75, axis=2)

plt.figure()
for i0 in range(nSeps):
    sep = seps[i0]
    plt.plot(samples, medianX[i0, :], color = colsSeps[i0], marker = 'o', markersize = 7, label = "Separation = " + str(sep))
    plt.fill_between(samples, p25X[i0, :], p75X[i0, :], color = colsSeps[i0], alpha = 0.2)
plt.xlabel("Number of samples")
plt.ylabel("Number of splits in x")
plt.grid(True)
plt.legend()
plt.savefig("tests/figures/nSplitsX_vs_samples.pdf")
plt.close()

plt.figure()
for i0 in range(nSeps):
    sep = seps[i0]
    plt.plot(samples, medianY[i0, :], color = colsSeps[i0], marker = 's', markersize = 7, label = "Separation = " + str(sep))
    plt.fill_between(samples, p25Y[i0, :], p75Y[i0, :], color = colsSeps[i0], alpha = 0.2)
plt.xlabel("Number of samples")
plt.ylabel("Number of splits in y")
plt.grid(True)
plt.legend()
plt.savefig("tests/figures/nSplitsY_vs_samples.pdf")
plt.close()
