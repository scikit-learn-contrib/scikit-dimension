# SPDX-License-Identifier: BSD-3-Clause
"""
Methods to embed a dataset into a higher dimensional space.

Adapted from the kmyim/scikit-dimension fork (John Harvey, Ka Man Yim) [IDkmyim]_.
"""

from itertools import combinations_with_replacement

import numpy as np


def _generate_monomials(n_variables, degree=5):
    """Exponent vectors of all monomials of exactly the given degree, shape (n_monomials, n_variables)."""
    return np.array(
        [
            np.bincount(c, minlength=n_variables)
            for c in combinations_with_replacement(range(n_variables), degree)
        ],
        dtype=int,
    ).reshape(-1, n_variables)


def monomial_embedding(data, degree=5):
    """
    Embed data into a higher dimensional space using all monomials of exactly the given degree.

    Parameters
    ----------
    data: np.array, (n_samples, n_features)
        Input data
    degree: int
        Degree of the monomials

    Returns
    -------
    data: np.array, (n_samples, n_monomials)
        Embedded data, with n_monomials = binom(n_features + degree - 1, degree)
    """
    data = np.asarray(data)
    monomials = _generate_monomials(data.shape[1], degree=degree)
    return np.prod(data[:, None, :] ** monomials[None, :, :], axis=2)


def delay_embedding(x, delay=range(-1, 2)):
    """
    Lift (time_steps, n_dim) time series data to (time_steps x (n_dim x n_delay)) data

    Parameters
    ----------
    x: np.array, (time_steps, n_dim)
        n_dim dimensional time series
    delay: integer or iterator of integers, such as list or one-dim numpy array
        if this is an integer, delay embedding is with this number of timesteps
        otherwise x[i] is embedded as [x[i+j] for j in delay]

    Returns
    -------
    data: np.array, (truncated time series length, n_dim x len(delay))
        Delay embedded time series, columns ordered by delay (sorted) then by dimension
    """
    delay = range(delay) if isinstance(delay, (int, np.integer)) else delay
    shifts = np.sort(np.asarray(list(delay))).astype(int)
    lead, lag = max(shifts[-1], 0), min(shifts[0], 0)
    trunc = lead - lag
    shifts -= lag  # shift must start at a non-negative number

    x = x.T  # n_dim x time_steps
    T = x.shape[-1]
    if T - trunc > 0:
        X = np.array(
            [x[:, d : T - (trunc - d)] for d in shifts]
        )  # n_feats x L x (T - trunc)
        return X.reshape([-1, T - trunc]).T
    else:
        raise ValueError("Delay embedding truncation exceeds length of time series.")
