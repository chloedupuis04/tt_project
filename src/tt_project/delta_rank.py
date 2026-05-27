import numpy as np

def delta_rank(Sigma,delta):
    d = len(Sigma)
    tail_sum = 0

    while d > 0 and tail_sum + Sigma[d-1]**2 <= delta**2:
        tail_sum += Sigma[d-1]**2
        d -= 1

    return d