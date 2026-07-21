# Here we want to test how consistent our partitions are

# Testing the greedy2_cross function (Python's version)

import numpy as np
from tt_project.greedy2_cross import *
from tt_project.adaptivity_in_n import *
from numpy.polynomial import chebyshev as cheb
from tt_project.interval_tree import IntervalTree
from tt_project.tt_to_tensor import tt_to_tensor
from matplotlib import pyplot as plt

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

def f2(x,y,z):
     return 1/((x-y)**2+(y-z)**2)

# We test for different number of y/x samples
nExp = 50 # Number of times we run each experiment
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
        find_subinterval_d(domain,n,axis,nbr_of_y_samples,f,x_root,depth=0,max_depth=100,tol=1e-6)
        x_intervals=x_root.intervals()
        nSplitsX[i1, i2] = len(x_intervals)
        # Splitting for y
        nbr_of_x_samples = samples[i1]
        axis=1
        y_root=IntervalTree(domain[axis, 0], domain[axis, 1])
        find_subinterval_d(domain,n,axis,nbr_of_x_samples,f,y_root,depth=0,max_depth=20,tol=1e-6)
        y_intervals=y_root.intervals()
        nSplitsY[i1, i2] = len(y_intervals)
        print("Number of splits in x: {}, Number of splits in y: {}".format(nSplitsX[i1, i2], nSplitsY[i1, i2]))

# Plot
median_x=np.median(nSplitsX, axis=1)
q25_x=np.percentile(nSplitsX, 25, axis=1)
q75_x=np.percentile(nSplitsX, 75, axis=1)

median_y=np.median(nSplitsY, axis=1)
q25_y=np.percentile(nSplitsY, 25, axis=1)
q75_y=np.percentile(nSplitsY, 75, axis=1)

#plot in x
plt.plot(samples, median_x, marker='o', label='Interval in x')
plt.fill_between(samples, q25_x, q75_x, alpha=0.25)
plt.xlabel("Number of samples")
plt.ylabel("Number of output intervals")
plt.title("Number of intervals versus number of samples in x")
plt.legend()
plt.grid(True)
plt.savefig("tests/figures/number_of_intervals_vs_samples_in_x.pdf")
plt.close()

#plot in y
plt.plot(samples, median_y, marker='o', label='Interval in y')
plt.fill_between(samples, q25_y, q75_y, alpha=0.25)
plt.xlabel("Number of samples")
plt.ylabel("Number of output intervals")
plt.title("Number of intervals versus number of samples in y")
plt.legend()
plt.grid(True)
plt.savefig("tests/figures/number_of_intervals_vs_samples_in_y.pdf")
plt.close()


mean_x=np.mean(nSplitsX, axis=1)
rel_iqr_x = (q75_x-q25_x) / mean_x
mean_y=np.mean(nSplitsY, axis=1)
rel_iqr_y = (q75_y-q25_y) / mean_y
for rel_iqr, label in [(rel_iqr_x, "x"), (rel_iqr_y, "y")]:
    below = rel_iqr < 0.10
    stable_idx = None
    for k in range(len(below)):
        if np.all(below[k:]):
            stable_idx = k
            break
    if stable_idx is not None:
        print(f"[{label}] splits become consistent (relative IQR < 10%, sustained) "
              f"from {samples[stable_idx]} samples onward.")
    else:
        print(f"[{label}] relative IQR never sustainably drops below 10% "
              f"over the tested range {samples[0]}-{samples[-1]}.")
        
print("test with 3D function")
nExp = 50 # Number of times we run each experiment
samples = [10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100] # Number of samples to test
nSamples = len(samples)
nSplitsX = np.zeros((nSamples, nExp), dtype=int) # Number of splits in x for each experiment
nSplitsY = np.zeros((nSamples, nExp), dtype=int) # Number of splits in x for each experiment
nSplitsZ=np.zeros((nSamples,nExp),dtype=int) # Number of splits in z for each experiment
n = 8 # Degree of local Chebyshev coefficient

for i1 in range(nSamples):
    for i2 in range(nExp):
        nbr_of_yz_samples = samples[i1]
        print("Experiment {} with {} samples".format(i2, nbr_of_yz_samples))

        # Start splitting for x
        domain=np.array([[0,1]]+[[1,2]] +[[1,2]], dtype=float)
        axis=0
        x_root = IntervalTree(domain[axis, 0], domain[axis, 1])
        find_subinterval_d(domain,n,axis,nbr_of_yz_samples,f2,x_root,depth=0,max_depth=100,tol=1e-6)
        x_intervals=x_root.intervals()
        nSplitsX[i1, i2] = len(x_intervals)

        # Splitting for y
        nbr_of_xz_samples = samples[i1]
        axis=1
        y_root=IntervalTree(domain[axis, 0], domain[axis, 1])
        find_subinterval_d(domain,n,axis,nbr_of_xz_samples,f2,y_root,depth=0,max_depth=20,tol=1e-6)
        y_intervals=y_root.intervals()
        nSplitsY[i1, i2] = len(y_intervals)

        # Splitting for z
        nbr_of_yz_samples = samples[i1]
        axis=2
        z_root = IntervalTree(domain[axis, 0], domain[axis, 1])
        find_subinterval_d(domain,n,axis,nbr_of_yz_samples,f2,z_root,depth=0,max_depth=100,tol=1e-6)
        z_intervals=z_root.intervals()
        nSplitsZ[i1, i2] = len(z_intervals)
        print("Number of splits in x: {}, Number of splits in y: {}, Number of splits in z: {}".format(nSplitsX[i1, i2], nSplitsY[i1, i2],nSplitsZ[i1,i2]))

# Plot
median_x=np.median(nSplitsX, axis=1)
q25_x=np.percentile(nSplitsX, 25, axis=1)
q75_x=np.percentile(nSplitsX, 75, axis=1)

median_y=np.median(nSplitsY, axis=1)
q25_y=np.percentile(nSplitsY, 25, axis=1)
q75_y=np.percentile(nSplitsY, 75, axis=1)

median_z=np.median(nSplitsZ, axis=1)
q25_z=np.percentile(nSplitsZ, 25, axis=1)
q75_z=np.percentile(nSplitsZ, 75, axis=1)


#plot in x
plt.plot(samples, median_x, marker='o', label='Interval in x')
plt.fill_between(samples, q25_x, q75_x, alpha=0.25)
plt.xlabel("Number of samples")
plt.ylabel("Number of output intervals")
plt.title("Number of intervals versus number of samples in x")
plt.legend()
plt.grid(True)
plt.savefig("tests/figures/number_of_intervals_vs_samples_in_x_f2.pdf")
plt.close()

#plot in y
plt.plot(samples, median_y, marker='o', label='Interval in y')
plt.fill_between(samples, q25_y, q75_y, alpha=0.25)
plt.xlabel("Number of samples")
plt.ylabel("Number of output intervals")
plt.title("Number of intervals versus number of samples in y")
plt.legend()
plt.grid(True)
plt.savefig("tests/figures/number_of_intervals_vs_samples_in_y_f2.pdf")
plt.close()

#plot in z
plt.plot(samples, median_z, marker='o', label='Interval in z')
plt.fill_between(samples, q25_z, q75_z, alpha=0.25)
plt.xlabel("Number of samples")
plt.ylabel("Number of output intervals")
plt.title("Number of intervals versus number of samples in z")
plt.legend()
plt.grid(True)
plt.savefig("tests/figures/number_of_intervals_vs_samples_in_z_f2.pdf")
plt.close()