"""
Methods to embed a dataset into a higher dimensional space.
"""

import numpy as np


def _generate_monomials(n_variables, degree=5):
    """Create a numpy array with the coefficients of all monomials."""

    if degree == 0:
        return np.zeros([n_variables], dtype=int)

    output = []
    initial_monomials = list(np.eye(n_variables, dtype=int))
    for x_var, initial_monomial in enumerate(initial_monomials):
        next_monomials = _generate_monomials(
            n_variables=n_variables - x_var, degree=degree - 1
        )

        befores = [0 for _ in range(next_monomials.ndim - 1)]
        befores.append(n_variables - next_monomials.shape[-1])
        afters = [0 for _ in range(next_monomials.ndim)]
        pad_size = tuple(zip(befores, afters))
        next_monomials = np.pad(next_monomials, pad_size, mode="constant")

        result = initial_monomial + next_monomials
        if result.ndim == 2:
            output = output + list(result)
        else:
            output.append(result)
    return np.stack(output)


def monomial_embedding(data, degree=5):
    """Embed 'data' into higher dimensional space using monomials."""
    input_dimension = np.shape(data)[1]
    monomials = _generate_monomials(input_dimension, degree=degree)
    return np.hstack(
        [
            np.product(np.power(data, monomial), axis=1, keepdims=True)
            for monomial in monomials
        ]
    )

def delay_embedding(x, delay = range(-1,2)):
    """
    Lift (time_steps, n_dim) time series data to (time_steps x (n_dim x n_delay)) data

    Parameters
    ----------
    x: np.array, (time_steps, n_dim) 
        n_dim dimensional time series
    delay: iterator of integers, such as list or one-dim numpy array
        delay embedding parameter such that x[i] is embedded as [x[i+j] for j in delay] 

    Returns
    -------
    data: np.array, (truncated time series length, n_dim x len(delay))
        Delay embedded time series
    """
    shifts = np.sort(delay).astype(int)
    lead,lag= max(shifts[-1],0), min(shifts[0],0)
    trunc = lead - lag
    shifts -= lag #shift must start at a non-negative number 

    x = x.T #n_dim x time_steps
    T = x.shape[-1]
    if T - trunc > 0:
        X = np.array([
            x[:,d:T-(trunc - d)]
            for d in shifts
        ])# n_feats x L x T'
        print()
        return X.reshape([-1, T-trunc]).T
    else:
        raise ValueError("Delay embedding truncation exceeds length of time series.")


def nonlinear():
    raise NotImplementedError


def linear():
    raise NotImplementedError


def radial_basis_function(data, n_functions=None):
    input_dimension = np.shape(data)[1]
    if not n_functions:
        n_functions = input_dimension
    landmarks = data[np.random.choice(data.shape[0], n_functions, replace=False), :]
    # TODO Define function using distances from landmarks

    raise NotImplementedError


def kernel_trick():
    raise NotImplementedError


def dictionary_learning():
    raise NotImplementedError


def tangent_function():
    raise NotImplementedError


