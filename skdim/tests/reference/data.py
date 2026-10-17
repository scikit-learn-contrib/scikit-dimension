# SPDX-License-Identifier: BSD-3-Clause
"""Deterministic inputs shared by generate_fixtures.py and test_reference.py.

RandomState streams are stable across numpy versions and the spike directions come from a
Hadamard basis (no BLAS-dependent QR), so the data can be rebuilt instead of stored."""
import numpy as np
from scipy.linalg import hadamard
from skdim.datasets import hyperBall, hyperSphere


def id_datasets():
    """Manifold samples for the ID estimators."""
    return {
        "ball5_in10": np.c_[hyperBall(300, 5, random_state=0), np.zeros((300, 5))],
        "sphere4_in5": hyperSphere(300, 4, random_state=1),
    }


def spiked(n, p, theta, seed):
    """n x p data with covariance I + U diag(theta) U^T, U the first columns of a normalized Hadamard matrix."""
    rng = np.random.RandomState(seed)
    ell = np.ones(p)
    ell[: len(theta)] += theta
    return (rng.randn(n, p) * np.sqrt(ell)) @ (hadamard(p) / np.sqrt(p))


def rank_datasets():
    """Spiked-covariance samples (noise variance 1) for the signal-rank versions of lPCA.
    Spikes are well above the BBP edge sqrt(p/n), so all methods should agree on them."""
    return {
        "n500_p128_r3": spiked(500, 128, [10.0, 6.0, 3.0], 0),
        "n200_p256_r2": spiked(200, 256, [20.0, 10.0], 1),
        # spikes near the BBP edge sqrt(p/n) = 0.71: exact agreement here tests the threshold conventions
        "edge_n256_p128": spiked(256, 128, [5.0, 1.0, 0.8, 0.6], 2),
    }


EASY = ("n500_p128_r3", "n200_p256_r2")  # where differently seeded permutation nulls must agree
