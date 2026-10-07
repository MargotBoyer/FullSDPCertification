"""Objective of the rank-r Burer-Monteiro factorized problem (bm_linf.m's myobj):
f(U) = Cost . U[:, 0] + Offset -- linear in the *first* column of U (the actual primal
point; the other r-1 columns are extra rank directions, see problem.py)."""
import numpy as np


def unpack(variables, nx, r):
    """variables: flat vector of length nx*r, Fortran order (column-major): the first
    nx entries are U[:, 0] (the point itself), matching BM-r's `[x; vec(V)]` stacking."""
    return variables.reshape((nx, r), order="F")


def pack(U):
    return U.reshape(-1, order="F")


def objective(variables, prob, r):
    U = unpack(variables, prob.nx, r)
    return float(prob.c @ U[:, 0] + prob.c0)


def gradient(variables, prob, r):
    df = np.zeros((prob.nx, r))
    df[:, 0] = prob.c
    return pack(df)
