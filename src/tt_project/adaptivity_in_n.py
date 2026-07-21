import numpy as np
from numpy.polynomial import chebyshev as cheb
from tt_project.interval_tree import IntervalTree


def make_1d_slice(f, fixed_sample, axis, dimension):

    fixed_sample = np.asarray(fixed_sample)


    def function_1d(variable):
        coordinates = []

        fixed_index = 0

        for current_axis in range(dimension):
            if current_axis == axis:
                coordinates.append(variable)
            else:
                coordinates.append(fixed_sample[fixed_index])
                fixed_index += 1

        return f(*coordinates)

    return function_1d

def find_subinterval(domain,n,axis,nbr_of_y_samples,f,node,depth,max_depth,tol=1e-6):
    
    resolved=[]
    other_axis=1-axis
    fixed_samples=np.random.uniform(domain[other_axis,0], domain[other_axis,1], size=nbr_of_y_samples)
    values_test = np.linspace(domain[axis,0], domain[axis,1], 1000,endpoint=True)
    for fixed_sample in fixed_samples:#this was not correct in the previous code 
        if axis == 0:
            function_1d = lambda x: f(x, fixed_sample) 
        elif axis == 1:
            function_1d = lambda y: f(fixed_sample, y)
        else:
            raise ValueError("For a 2D function, axis must be 0 or 1.")

        p = cheb.Chebyshev.interpolate(function_1d,deg=n - 1,domain=domain[axis])
    
        f_values = function_1d(values_test)
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

def find_subinterval_d(domain,n,axis,nbr_of_samples,f,node,depth,max_depth,tol=1e-6):
    
    resolved=[]
    d=domain.shape[0]

    other_axes = [ax for ax in range(d) if ax != axis]
    fixed_samples_mat = np.zeros((d - 1, nbr_of_samples))

    for idx, other_axis in enumerate(other_axes): 
        fixed_samples_mat[idx, :] = np.random.uniform(domain[other_axis, 0], domain[other_axis, 1], size=nbr_of_samples)

    values_test = np.linspace(domain[axis,0], domain[axis,1], 1000,endpoint=True)
    
    for j in range(nbr_of_samples):
        fixed_variables = fixed_samples_mat[:,j]
        function_1d = make_1d_slice(f,fixed_variables,axis,d)
        p= cheb.Chebyshev.interpolate(function_1d,deg=n - 1,domain=domain[axis])
        f_values = function_1d(values_test)
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
    find_subinterval_d(left_domain,n,axis,nbr_of_samples,f,left_node,depth=depth+1,max_depth=max_depth,tol=tol)
    find_subinterval_d(right_domain,n,axis,nbr_of_samples,f,right_node,depth=depth+1,max_depth=max_depth,tol=tol)

