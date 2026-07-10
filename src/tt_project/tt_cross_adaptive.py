import numpy as np
from tensorly.contrib.decomposition import tensor_train_cross as _tensorly_tensor_train_cross
from tt_project.tt_to_tensor import tt_to_tensor


def tensor_train_cross(input_tensor, rank, tol=1e-6, cross_tol=1e-10, n_iter_max=100, random_state=None, verbose=False):
    """Rank-adaptive drop-in replacement for
    tensorly.contrib.decomposition.tensor_train_cross.

    tensorly's tensor_train_cross documents `rank` as a "maximum allowable
    rank", but its implementation never truncates: every core is built at
    exactly the requested rank and `tol` only bounds convergence of its own
    internal ALS sweeps, not the accuracy of the returned decomposition
    against the true tensor. So it always returns the maximum rank you give
    it, regardless of `tol`.

    This wrapper instead sweeps candidate ranks from 1 up to the requested
    maximum, calling tensorly's tensor_train_cross at each one and measuring
    the true relative error against input_tensor. It stops as soon as `tol`
    is met or the maximum rank is reached, whichever happens first.

    Parameters
    ----------
    input_tensor : ndarray
    rank : int or list of int
        Maximum allowable TT rank(s), same convention as tensorly's `rank`:
        an int applies to every internal bond, or a list of length ndim+1
        with rank[0] == rank[-1] == 1 caps each bond individually.
    tol : float
        Desired relative error ||input_tensor - TT|| / ||input_tensor||.
    cross_tol : float
        Convergence tolerance for tensorly's own ALS sweep at each candidate
        rank. This is not the accuracy target (see above) -- it just needs
        to be tight enough that tensorly's cross iteration fully converges
        before we measure the true error at that rank.
    n_iter_max : int
        Passed through to tensorly's tensor_train_cross.
    random_state : optional
    verbose : bool
        Print the rank and true error tried at each step.

    Returns
    -------
    cores : list of ndarray
        TT cores meeting either `tol` or the requested maximum rank.
    """
    input_tensor = np.asarray(input_tensor)
    order = input_tensor.ndim

    if isinstance(rank, int):
        max_ranks = [1] + [rank] * (order - 1) + [1]
    else:
        max_ranks = list(rank)

    ceiling = max(max_ranks[1:-1]) if order > 1 else 1
    norm_tensor = np.linalg.norm(input_tensor)

    cores = None
    for r in range(1, ceiling + 1):
        trial_rank = [1] + [min(r, max_ranks[k]) for k in range(1, order)] + [1]
        at_ceiling = all(trial_rank[k] == max_ranks[k] for k in range(1, order))

        try:
            cores = _tensorly_tensor_train_cross(
                input_tensor, rank=trial_rank, tol=cross_tol,
                n_iter_max=n_iter_max, random_state=random_state,
            )
        except ValueError as exc:
            if verbose:
                print(f"[tensor_train_cross adaptive] rank={trial_rank} failed to converge ({exc})")
            if at_ceiling:
                raise
            continue

        error = np.linalg.norm(input_tensor - tt_to_tensor(cores)) / norm_tensor
        if verbose:
            print(f"[tensor_train_cross adaptive] rank={trial_rank} error={error:.3e}")

        if error <= tol or at_ceiling:
            break

    return cores
