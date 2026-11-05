# SPDX-License-Identifier: BSD-3-Clause AND MIT
# hyperTwinPeaks, swissRoll3Sph: adapted from R intrinsicDimension, Copyright (c) 2017 Kerstin Johnsson, MIT.
# BenchmarkManifolds: adapted from https://github.com/stat-ml/GeoMLE, Copyright (c) 2019 Mokrov Nikita,
#   Marina Gomtsyan, Maxim Panov and Yury Yanovich, MIT.
# lorenz_attractor: adapted from https://github.com/scipython/scipython-maths, Copyright (c) 2021 Christian Hill, MIT.
# lineDiskBall: adapted from R package ider by Hideitsu Hino, originally GPL-2.
# product, torus, toroidal_spiral, dumbbell, ...: from the kmyim/scikit-dimension fork (John Harvey, Ka Man Yim).
# HomID generators: adapted from https://github.com/a-writes-code/homid,
#   Copyright (c) 2026 a-writes-code, MIT.
# See LICENSES_bundled.txt and AUTHORS.md
from itertools import combinations, islice
from math import comb
from numbers import Integral
import warnings

import numpy as np
from sklearn.utils.validation import check_random_state
from scipy.special import gammainc
from scipy.integrate import solve_ivp
from scipy.stats import ortho_group


def product(n, sample_1, sample_1_kwargs=None, sample_2=None, sample_2_kwargs=None):
    """
    Generates a sample from the product of two datasets of this module, by stacking their coordinates.
    Adapted from the kmyim/scikit-dimension fork (John Harvey) [IDkmyim]_.

    Parameters
    ----------
    n: int
        Number of data points.
    sample_1: callable
        Generator of the first factor, called as sample_1(n=n, **sample_1_kwargs)
    sample_1_kwargs: dict, optional
        Keyword arguments passed to sample_1
    sample_2: callable, optional
        Generator of the second factor. Defaults to sample_1
    sample_2_kwargs: dict, optional
        Keyword arguments passed to sample_2. Pass different random_state values to
        the two factors, otherwise their coordinates may be correlated.

    Returns
    -------
    data: np.array, (npoints x ndim)
        Generated data, the columns of sample_1 followed by those of sample_2
    """
    if sample_2 is None:
        sample_2 = sample_1
    return np.hstack(
        (
            sample_1(n=n, **(sample_1_kwargs or {})),
            sample_2(n=n, **(sample_2_kwargs or {})),
        )
    )


def hyperBall(n, d, radius=1.0, center=[], random_state=None):
    """
    Generates a sample from a uniform distribution on the hyperball

    Parameters
    ----------
    n: int 
        Number of data points.
    d: int
        Dimension of the hyperball
    radius: float
        Radius of the hyperball
    center: list, tuple, np.array
        Center of the hyperball
    random_state: int, np.random.RandomState instance
        Random number generator

    Returns
    -------
    data: np.array, (npoints x ndim)
        Generated data
    """
    random_state_ = check_random_state(random_state)
    if center == []:
        center = np.array([0] * d)
    r = radius
    x = random_state_.normal(size=(n, d))
    ssq = np.sum(x ** 2, axis=1)
    fr = r * gammainc(d / 2, ssq / 2) ** (1 / d) / np.sqrt(ssq)
    frtiled = np.tile(fr.reshape(n, 1), (1, d))
    p = center + np.multiply(x, frtiled)
    return p


def hyperSphere(n, d, random_state=None, radius=1.0, center=None):
    """
    Generates a sample from a uniform distribution on the hypersphere

    Parameters
    ----------
    n: int
        Number of data points.
    d: int
        Dimension of the hypersphere
    random_state: int, np.random.RandomState instance
        Random number generator
    radius: float
        Radius of the hypersphere
    center: list, tuple, np.array, optional
        Center of the hypersphere, of length d+1. Defaults to the origin

    Returns
    -------
    data: np.array, (npoints x ndim)
        Generated data
    """
    random_state = check_random_state(random_state)
    vec = random_state.randn(n, d + 1)
    vec /= np.linalg.norm(vec, axis=1)[:, None]
    if center is None:
        center = np.zeros(d + 1)
    return np.asarray(center) + radius * vec


def hyperTwinPeaks(n, d=2, height=1.0, random_state=None):
    """ 
    Generates a sample from a plane with protruding peaks. Translated from Kerstin Johnsson's R package intrinsicDimension

    Parameters
    ----------
    n: int 
        Number of data points.
    d: int
        Dimension of the dataset
    height: float
        Height of the peaks
    random_state: int, np.random.RandomState instance
        Random number generator

    Returns
    -------
    data: np.array, (npoints x ndim)
        Generated data
    """

    random_state = check_random_state(random_state)
    base_coord = random_state.uniform(size=(n, d))
    _height = height * np.prod(np.sin(2 * np.pi * base_coord), axis=1, keepdims=1)
    return np.hstack((base_coord, _height))


def hyperTwinPeaksUniform(n, d=2, height=1.0, random_state=None):
    """
    Generates a sample from the hyperTwinPeaks surface that is uniform with respect to its surface measure,
    by rejection sampling on the volume element sqrt(1 + |grad z|^2) of z = height * prod_i sin(2 pi x_i).
    Adapted from the kmyim/scikit-dimension fork (John Harvey) [IDkmyim]_.

    Parameters
    ----------
    n: int
        Number of data points.
    d: int
        Dimension of the dataset
    height: float
        Height of the peaks
    random_state: int, np.random.RandomState instance
        Random number generator

    Returns
    -------
    data: np.array, (npoints x ndim)
        Generated data
    """
    random_state = check_random_state(random_state)
    # sum_j cos_j^2 prod_{i!=j} sin_i^2 <= 1, so the volume element is bounded by:
    vol_max = np.sqrt(1 + (2 * np.pi * height) ** 2)
    data = np.empty((0, d + 1))
    while data.shape[0] < n:
        x = random_state.uniform(size=(n, d))
        s, c = np.sin(2 * np.pi * x), np.cos(2 * np.pi * x)
        grad = np.stack(
            [
                2 * np.pi * height * c[:, j] * np.prod(np.delete(s, j, axis=1), axis=1)
                for j in range(d)
            ],
            axis=1,
        )
        vol = np.sqrt(1 + np.sum(grad**2, axis=1))
        keep = random_state.uniform(0, vol_max, n) < vol
        z = height * np.prod(s[keep], axis=1, keepdims=True)
        data = np.vstack((data, np.hstack((x[keep], z))))
    return data[:n]


def lineDiskBall(n, random_state=None):
    """ 
    Generates a sample from a uniform distribution on a line, an oblong disk and an oblong ball
    Translated from the ldbl function of Hideitsu Hino's R package ider [IDHino]_.

    Parameters
    ----------
    n: int 
        Number of data points.
    random_state: int, np.random.RandomState instance
        Random number generator

    Returns
    -------
    labels: np.array, (npoints x 1)
        Labels
    data: np.array, (npoints x ndim)
        Generated data

    """

    random_state = check_random_state(random_state)

    line = np.hstack(
        (
            np.repeat(0, 5 * n)[:, None],
            np.repeat(0, 5 * n)[:, None],
            random_state.uniform(-0.5, 0, size=5 * n)[:, None],
        )
    )
    disc = np.hstack(
        (random_state.uniform(-1, 1, (13 * n, 2)), np.zeros(13 * n)[:, None],)
    )
    disc = disc[~(np.sqrt(np.sum(disc ** 2, axis=1)) > 1), :]
    disc = disc[:, [0, 2, 1]]
    disc[:, 2] = disc[:, 2] - min(disc[:, 2]) + max(line[:, 2])

    fb = random_state.uniform(-0.5, 0.5, size=(n * 100, 3))
    rmID = np.where(np.sqrt(np.sum(fb ** 2, axis=1)) > 0.5)[0]

    if len(rmID) > 0:
        fb = fb[~(np.sqrt(np.sum(fb ** 2, axis=1)) > 0.5), :]

    fb = np.hstack((fb[:, :2], fb[:, [2]] + 0.5))
    fb[:, 2] = fb[:, 2] - min(fb[:, 2]) + max(disc[:, 2])

    #     if _sorted:
    #         fb = fb[order(fb[:, 2]),:]

    line2 = np.hstack(
        (
            np.repeat(0, 5 * n)[:, None],
            np.repeat(0, 5 * n)[:, None],
            random_state.uniform(-0.5, 0, size=5 * n)[:, None],
        )
    )
    line2[:, 2] = line2[:, 2] - min(line2[:, 2]) + max(fb[:, 2])
    lineID = np.repeat(1, len(line))
    discID = np.repeat(2, len(disc))
    fbID = np.repeat(3, len(fb))
    line2ID = np.repeat(1, len(line2))
    x = np.vstack((line, disc, fb, line2))
    useID = np.sort(random_state.choice(len(x), n, replace=False))
    x = x[useID, :]

    return x, np.concatenate((lineID, discID, fbID, line2ID), axis=0)[useID]


def swissRoll3Sph(n_swiss, n_sphere, a=1, b=2, nturn=1.5, h=4, random_state=None):
    """
    Generates a sample from a uniform distribution on a Swiss roll-surface, 
    possibly together with a sample from a uniform distribution on a 3-sphere
    inside the Swiss roll. Translated from Kerstin Johnsson's R package intrinsicDimension

    Parameters
    ----------
    n_swiss: int 
        Number of data points on the Swiss roll.
    n_sphere: int
        Number of data points on the 3-sphere.
    a: int or float, default=1
        Minimal radius of Swiss roll and radius of 3-sphere.
    b: int or float, default=2
        Maximal radius of Swiss roll.
    nturn: int or float, default=1.5
        Number of turns of the surface. 
    h: int or float, default=4
        Height of Swiss roll.

    Returns
    -------
    data: np.array, (npoints x ndim)
        Generated data
    """
    random_state = check_random_state(random_state)

    if n_swiss > 0:
        omega = 2 * np.pi * nturn

        def dl(r):
            return np.sqrt(b ** 2 + omega ** 2 * (a + b * r) ** 2)

        ok = np.zeros(1)
        while sum(ok) < n_swiss:
            r_samp = random_state.uniform(size=3 * n_swiss)
            ok = random_state.uniform(size=3 * n_swiss) < dl(r_samp) / dl(1)

        r_samp = r_samp[ok][:n_swiss]
        x = (a + b * r_samp) * np.cos(omega * r_samp)
        y = (a + b * r_samp) * np.sin(omega * r_samp)
        z = random_state.uniform(-h, h, size=n_swiss)
        w = np.zeros(n_swiss)

    else:
        x = y = z = w = np.array([])

    if n_sphere > 0:
        sph = hyperSphere(n_sphere, 4, random_state=random_state) * a
        x = np.concatenate((x, sph[:, 0]))
        y = np.concatenate((y, sph[:, 1]))
        z = np.concatenate((z, sph[:, 2]))
        w = np.concatenate((w, sph[:, 3]))

    return np.hstack((x[:, None], y[:, None], z[:, None], w[:, None]))


def toroidal_spiral(n, n_twists=20, r1=1.0, r2=0.25, random_state=None):
    """
    Generates a sample from a uniform distribution (with respect to arc length) on a curve
    winding n_twists times around a torus in R^3.
    Adapted from the kmyim/scikit-dimension fork (John Harvey, Ka Man Yim) [IDkmyim]_.

    Parameters
    ----------
    n: int
        Number of data points.
    n_twists: int
        Number of turns around the tube of the torus
    r1: float
        Radius of the central circle
    r2: float
        Radius of the tube, must satisfy r2 < r1
    random_state: int, np.random.RandomState instance
        Random number generator

    Returns
    -------
    data: np.array, (npoints x 3)
        Generated data
    """
    if not r1 > r2:
        raise ValueError("The radii must satisfy r1 > r2.")
    random_state = check_random_state(random_state)
    # speed |gamma'(phi)| = sqrt((r1 + r2 cos(k phi))^2 + (k r2)^2)
    k = n_twists
    speed_max = np.sqrt((r1 + r2) ** 2 + (k * r2) ** 2)
    phi = np.empty(0)
    while phi.shape[0] < n:
        p = random_state.uniform(0, 2 * np.pi, n)
        speed = np.sqrt((r1 + r2 * np.cos(k * p)) ** 2 + (k * r2) ** 2)
        phi = np.concatenate((phi, p[random_state.uniform(0, speed_max, n) < speed]))
    phi = phi[:n]
    rho = r1 + r2 * np.cos(k * phi)
    return np.stack((rho * np.cos(phi), rho * np.sin(phi), r2 * np.sin(k * phi)), axis=1)


def torus(n, r1=1.0, r2=0.25, random_state=None):
    """
    Generates a sample from a uniform distribution on the 2d torus in R^3.
    Adapted from the kmyim/scikit-dimension fork (John Harvey, Ka Man Yim) [IDkmyim]_.

    Parameters
    ----------
    n: int
        Number of data points.
    r1: float
        Radius of the central circle
    r2: float
        Radius of the tube, must satisfy r2 < r1
    random_state: int, np.random.RandomState instance
        Random number generator

    Returns
    -------
    data: np.array, (npoints x 3)
        Generated data
    """
    if not r1 > r2:
        raise ValueError("The radii must satisfy r1 > r2.")
    random_state = check_random_state(random_state)
    phi, theta = np.empty(0), np.empty(0)
    while phi.shape[0] < n:
        p = random_state.uniform(0, 2 * np.pi, n)
        t = random_state.uniform(0, 2 * np.pi, n)
        # area element (r1 + r2 cos(theta)) dphi dtheta
        keep = random_state.uniform(0, r1 + r2, n) < r1 + r2 * np.cos(t)
        phi, theta = np.concatenate((phi, p[keep])), np.concatenate((theta, t[keep]))
    phi, theta = phi[:n], theta[:n]
    rho = r1 + r2 * np.cos(theta)
    return np.stack((rho * np.cos(phi), rho * np.sin(phi), r2 * np.sin(theta)), axis=1)


def lorenz_attractor(n=10000, tmax=100.0):
    """
    Generates a deterministic sample from the Lorenz attractor (sigma=10, beta=2.667, rho=28),
    by integrating the Lorenz equations from (0, 1, 1.05) and evaluating the trajectory at n
    evenly spaced times in [0, tmax]. The initial transient is included.
    Adapted from the kmyim/scikit-dimension fork (John Harvey, Ka Man Yim) [IDkmyim]_.

    Parameters
    ----------
    n: int
        Number of data points.
    tmax: float
        Integration time

    Returns
    -------
    data: np.array, (npoints x 3)
        Generated data
    """
    # Adapted from lorenz.py by Christian Hill (2016, 2021),
    # https://github.com/scipython/scipython-maths, MIT License; see LICENSES_bundled.txt
    sigma, beta, rho = 10, 2.667, 28
    u0, v0, w0 = 0, 1, 1.05

    def lorenz(t, X, sigma, beta, rho):
        """The Lorenz equations."""
        u, v, w = X
        up = -sigma * (u - v)
        vp = rho * u - v - u * w
        wp = -beta * w + u * v
        return up, vp, wp

    soln = solve_ivp(
        lorenz,
        (0, tmax),
        (u0, v0, w0),
        t_eval=np.linspace(0, tmax, n),
        args=(sigma, beta, rho),
    )
    return soln.y.T


def dumbbell(n, connecting_radius=0.1, random_state=None):
    """
    Generates a sample from a uniform distribution on a piecewise-linear 2d dumbbell surface in R^3:
    the surface of revolution around the x-axis of a profile on [-1, 1] that rises linearly from 0
    to 0.5 at the ends and narrows to a tube of radius connecting_radius on [-0.2, 0.2].
    Adapted from the kmyim/scikit-dimension fork (John Harvey, Ka Man Yim) [IDkmyim]_.

    Parameters
    ----------
    n: int
        Number of data points.
    connecting_radius: float
        Radius of the connecting tube
    random_state: int, np.random.RandomState instance
        Random number generator

    Returns
    -------
    data: np.array, (npoints x 3)
        Generated data
    """
    # profile breakpoints: radius 0 at +-1, 0.5 at +-0.6, connecting_radius on [-0.2, 0.2]
    def profile_function(x):
        return np.interp(
            x,
            [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0],
            [0.0, 0.5, connecting_radius, connecting_radius, 0.5, 0.0],
        )

    return SurfaceOfRevolution(profile_function, random_state=random_state).generate(n=n)


_HOMID_TEMP_BYTES = 16 * 1024**2


def _homid_integer(value, name, minimum):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < minimum
    ):
        raise ValueError(
            f"{name} must be an integer greater than or equal to {minimum}."
        )
    return int(value)


def _homid_parameters(n, p, k):
    n = _homid_integer(n, "n", 0)
    p = _homid_integer(p, "p", 2)
    k = _homid_integer(k, "k", 1)
    if k >= p:
        raise ValueError("k must be smaller than p.")
    return n, p, k


def _homid_empty(n, n_features):
    max_size = np.iinfo(np.intp).max
    if n_features > max_size or (n and n_features > max_size // 8 // n):
        raise ValueError(
            "The requested dataset is too large for a NumPy array."
        )
    if n and (n_features >= 10_000 or n * n_features * 8 >= 256 * 1024**2):
        warnings.warn(
            "Large HomID dataset: generation may be slow and the dense output "
            f"requires {n * n_features * 8 / 1024**2:.1f} MiB. "
            "Vector feature counts grow combinatorially. Consider smaller "
            "parameters and cache repeated requests locally.",
            UserWarning,
            stacklevel=3,
        )
    return np.empty((n, n_features), dtype=float)


def _homid_blocks(n, row_size):
    batch_size = max(1, min(256, _HOMID_TEMP_BYTES // (8 * row_size)))
    for start in range(0, n, batch_size):
        yield slice(start, min(start + batch_size, n))


def _homid_replay_seed(random_state, n):
    random_state_instance = check_random_state(random_state)
    if n > 2**32:
        raise ValueError("The per-point seed sequence exceeds uint32 range.")
    if isinstance(random_state, Integral):
        seed = int(random_state)
    elif n:
        seed = int(random_state_instance.randint(
            0, 2**32 - n + 1, dtype=np.uint32
        ))
    else:
        return 0
    if n and seed + n - 1 >= 2**32:
        raise ValueError("The per-point seed sequence exceeds uint32 range.")
    return seed


def _homid_minors(frames, out):
    batch, p, k = frames.shape
    # Bound the temporary (batch, n_minors, k, k) array, not just samples.
    block_size = max(1, _HOMID_TEMP_BYTES // (8 * batch * k * k))
    rows = combinations(range(p), k)
    start = 0
    while indices := list(islice(rows, block_size)):
        stop = start + len(indices)
        out[:, start:stop] = np.linalg.det(frames[:, np.asarray(indices), :])
        start = stop


def grassmannian_projector(n, p, k, random_state=None):
    """
    Generate the HomID Grassmannian projector dataset [Das2025]_.

    Parameters
    ----------
    n: int
        Number of data points. Must be nonnegative.
    p: int
        Dimension of the underlying real vector space. Must be at least 2.
    k: int
        Subspace dimension, with 1 <= k < p.
    random_state: int, np.random.RandomState instance or None, optional
        Random number generator, as in the other dataset functions.

    Returns
    -------
    data: np.array, (n, p * p)
        Projectors flattened in row-major order. Intrinsic dimension k*(p-k).

    Warns
    -----
    UserWarning
        For nonempty output with at least 10000 features or 256 MiB of data.
        Large requests may be slow and require substantial memory.

    Notes
    -----
    Integer seeds replay the original HomID samples in the same numerical
    environment. Sampling uses full p-by-p Gaussian matrices and raw QR.
    The original Q @ D @ Q.T arithmetic is retained for rank-k diagonal D.
    Generation writes no files; see the HomID guide for optional local caching.

    Examples
    --------
    >>> grassmannian_projector(100, 4, 2, random_state=0).shape
    (100, 16)
    """
    n, p, k = _homid_parameters(n, p, k)
    random_state = check_random_state(random_state)
    data = _homid_empty(n, p * p)
    diagonal = np.zeros((p, p))
    np.fill_diagonal(diagonal[:k, :k], 1)
    for i in range(n):
        frame, _ = np.linalg.qr(random_state.randn(p, p))
        data[i] = (frame @ diagonal @ frame.T).ravel()
    return data


def grassmannian_vector(n, p, k, random_state=None):
    """
    Generate the HomID Grassmannian minor-vector dataset [Das2025]_.

    Parameters
    ----------
    n: int
        Number of data points. Must be nonnegative.
    p: int
        Dimension of the underlying real vector space. Must be at least 2.
    k: int
        Subspace dimension, with 1 <= k < p.
    random_state: int, np.random.RandomState instance or None, optional
        Random number generator, as in the other dataset functions.

    Returns
    -------
    data: np.array, (n, binom(p, k))
        Signed minor vectors. Intrinsic dimension k*(p-k).

    Warns
    -----
    UserWarning
        For nonempty output with at least 10000 features or 256 MiB of data.
        The feature count grows combinatorially with p and k.

    Notes
    -----
    Coordinates are the k-by-k minors of a Haar orthonormal p-by-k frame.
    Row-index combinations are lexicographically ordered. Signs are retained:
    these unit decomposable vectors describe oriented subspaces.
    An integer seed uses the original per-point seeds seed+i and full O(p)
    draws. RandomState instances and None instead draw one valid base seed.
    Empty requests consume no random draws. Exact replay requires the same
    NumPy, SciPy, and numerical backend as the original call.
    Determinants are batched and temporary arrays are chunked. The dense
    output still requires 8*n*binom(p, k) bytes. See the HomID guide for
    optional local caching.

    Examples
    --------
    >>> grassmannian_vector(100, 4, 2, random_state=0).shape
    (100, 6)
    """
    n, p, k = _homid_parameters(n, p, k)
    seed = _homid_replay_seed(random_state, n)
    data = _homid_empty(n, comb(p, k))
    for block in _homid_blocks(n, 3 * p * k + k * k):
        batch = block.stop - block.start
        frames = np.empty((batch, p, k))
        for j, i in enumerate(range(block.start, block.stop)):
            point_state = np.random.RandomState(seed + i)
            frames[j] = ortho_group.rvs(dim=p, random_state=point_state)[:, :k]
        _homid_minors(frames, data[block])
    return data


def stiefel_matrix(n, p, k, random_state=None):
    """
    Generate the HomID Stiefel matrix dataset [Das2025]_.

    Parameters
    ----------
    n: int
        Number of data points. Must be nonnegative.
    p: int
        Dimension of the underlying real vector space. Must be at least 2.
    k: int
        Number of orthonormal columns, with 1 <= k < p.
    random_state: int, np.random.RandomState instance or None, optional
        Random number generator, as in the other dataset functions.

    Returns
    -------
    data: np.array, (n, p * k)
        Frames flattened in row-major order. Intrinsic dimension
        p*k-k*(k+1)/2.

    Warns
    -----
    UserWarning
        For nonempty output with at least 10000 features or 256 MiB of data.
        Large requests may be slow and require substantial memory.

    Notes
    -----
    Frames use NumPy's raw Gaussian QR convention, without diagonal sign
    correction, as in the HomID research scripts. They are orthonormal, but
    this convention does not give invariant sampling on the full Stiefel
    manifold.
    Full p-by-p Gaussian draws retain the original integer-seed sequence.
    Generation writes no files; see the HomID guide for optional local caching.

    Examples
    --------
    >>> stiefel_matrix(100, 4, 2, random_state=0).shape
    (100, 8)
    """
    n, p, k = _homid_parameters(n, p, k)
    random_state = check_random_state(random_state)
    data = _homid_empty(n, p * k)
    for i in range(n):
        frame, _ = np.linalg.qr(random_state.randn(p, p))
        data[i] = frame[:, :k].ravel()
    return data


def stiefel_vector(n, p, k, random_state=None):
    """
    Generate the HomID Stiefel vector construction [Das2025]_.

    Parameters
    ----------
    n: int
        Number of data points. Must be nonnegative.
    p: int
        Dimension of the underlying real vector space. Must be at least 2.
    k: int
        Subspace dimension, with 1 <= k < p.
    random_state: int, np.random.RandomState instance or None, optional
        Random number generator, as in the other dataset functions.

    Returns
    -------
    data: np.array, (n, binom(p, k) + k * k)
        A minor vector concatenated with a row-major O(k) matrix.
        Intrinsic dimension p*k-k*(k+1)/2.

    Warns
    -----
    UserWarning
        For nonempty output with at least 10000 features or 256 MiB of data.
        The minor-vector feature count grows combinatorially with p and k.

    Notes
    -----
    This follows the HomID research implementation: an oriented Grassmannian
    minor vector and an independent Haar O(k) matrix. Both determinant
    components are included, rather than only the SO(k) factor in the
    paper's theoretical recipe.
    Its support is a product with the same local dimension as the Stiefel
    manifold; this recipe is not generally a global Stiefel embedding.
    For k=1 the O(1) factor is equally likely to be -1 or 1. The squared norm
    is 1+k.
    An integer seed replays per-point seeds seed+i. Each O(k) draw continues
    the same point's stream after the full O(p) draw. RandomState instances
    and None draw one valid base seed. Empty requests consume no random draws.
    Exact replay requires the same numerical environment as the original.
    See the HomID guide for optional local caching.

    Examples
    --------
    >>> stiefel_vector(100, 4, 2, random_state=0).shape
    (100, 10)
    """
    n, p, k = _homid_parameters(n, p, k)
    seed = _homid_replay_seed(random_state, n)
    n_minors = comb(p, k)
    data = _homid_empty(n, n_minors + k * k)
    for block in _homid_blocks(n, 4 * p * k + 6 * k * k):
        batch = block.stop - block.start
        frames = np.empty((batch, p, k))
        rotations = np.empty((batch, k, k))
        for j, i in enumerate(range(block.start, block.stop)):
            point_state = np.random.RandomState(seed + i)
            frames[j] = ortho_group.rvs(dim=p, random_state=point_state)[:, :k]
            rotations[j] = ortho_group.rvs(dim=k, random_state=point_state)
        _homid_minors(frames, data[block, :n_minors])
        data[block, n_minors:] = rotations.reshape(batch, k * k)
    return data


def flag_vector(n, p, k1, k2, random_state=None):
    """
    Generate the HomID flag minor-tensor dataset [Das2025]_.

    Parameters
    ----------
    n: int
        Number of data points. Must be nonnegative.
    p: int
        Dimension of the underlying real vector space. Must be at least 2.
    k1, k2: int
        Ranks of two nested column prefixes. Both must be positive and must
        not exceed p. They may coincide; their sum is not constrained.
    random_state: int, np.random.RandomState instance or None, optional
        Random number generator, as in the other dataset functions.

    Returns
    -------
    data: np.array, (n, binom(p, k1) * binom(p, k2))
        Row-major flattened outer products of the two minor vectors.
        Intrinsic dimension r*(p-r)+s*(r-s), where r=max(k1, k2) and
        s=min(k1, k2).

    Warns
    -----
    UserWarning
        For nonempty output with at least 10000 features or 256 MiB of data.
        The feature count grows as a product of two binomial coefficients.

    Notes
    -----
    The factors use columns [:k1] and [:k2] of the same Haar O(p) matrix,
    following the nested-prefix convention in the HomID research scripts.
    Minor indices within each factor are lexicographically ordered. For
    equal ranks, the output is the self-outer product of a unit minor vector,
    with intrinsic dimension k1*(p-k1).
    An integer seed uses per-point seeds seed+i and full O(p) draws, as in
    HomID. RandomState instances and None draw one valid base seed. Empty
    requests consume no random draws. Exact replay requires the same
    numerical environment as the original call.
    Each minor is computed once before forming the outer product. Temporary
    arrays are chunked, but the dense output can still be large. See the
    HomID guide for optional local caching.

    Examples
    --------
    >>> flag_vector(100, 4, 2, 2, random_state=0).shape
    (100, 36)
    """
    n = _homid_integer(n, "n", 0)
    p = _homid_integer(p, "p", 2)
    k1 = _homid_integer(k1, "k1", 1)
    k2 = _homid_integer(k2, "k2", 1)
    if k1 > p or k2 > p:
        raise ValueError("k1 and k2 must not exceed p.")
    seed = _homid_replay_seed(random_state, n)
    dim1, dim2 = comb(p, k1), comb(p, k2)
    data = _homid_empty(n, dim1 * dim2)
    width = max(k1, k2)
    row_size = 3 * p * width + width * width + dim1 + dim2
    for block in _homid_blocks(n, row_size):
        batch = block.stop - block.start
        frames = np.empty((batch, p, width))
        for j, i in enumerate(range(block.start, block.stop)):
            point_state = np.random.RandomState(seed + i)
            frames[j] = ortho_group.rvs(
                dim=p, random_state=point_state
            )[:, :width]
        first = np.empty((batch, dim1))
        second = first if k1 == k2 else np.empty((batch, dim2))
        _homid_minors(frames[:, :, :k1], first)
        if k1 != k2:
            _homid_minors(frames[:, :, :k2], second)
        np.multiply(
            first[:, :, None], second[:, None, :],
            out=data[block].reshape(batch, dim1, dim2),
        )
    return data


class BenchmarkManifolds:
    """
    Generates a commonly used benchmark set of synthetic manifolds with known intrinsic dimension described by Hein et al. and Campadelli et al. [Campadelli2015]_ Adapted from GeoMLE [IDGeoMLE]_.

    Parameters
    ----------
    noise_type : str, 'uniform' or 'gaussian'
        Type of noise to generate
    """

    # Adapted from https://github.com/stat-ml/GeoMLE (MIT); notice in LICENSES_bundled.txt

    def __init__(self, random_state: int = None, noise_type: str = "uniform"):

        self.random_state = check_random_state(random_state)
        self.noise_type = noise_type

        self._dict_truth = {
            "M1_Sphere": (10, 11, "10D sphere linearly embedded"),
            "M2_Affine_3to5": (3, 5, "Affine space"),
            "M3_Nonlinear_4to6": (
                4,
                6,
                "Concentrated figure, mistakable with a 3D one",
            ),
            "M4_Nonlinear": (4, 8, "Nonlinear manifold"),
            "M5a_Helix1d": (1, 3, "1D helix"),
            "M5b_Helix2d": (2, 3, "2D helix"),
            "M6_Nonlinear": (6, 36, "Nonlinear manifold"),
            "M7_Roll": (2, 3, "Swiss Roll"),
            "M8_Nonlinear": (12, 72, "Nonlinear (highly curved) manifold"),
            "M9_Affine": (20, 20, "Affine space"),
            "M10a_Cubic": (10, 11, "10D hypercube"),
            "M10b_Cubic": (17, 18, "17D hypercube"),
            "M10c_Cubic": (24, 25, "24D hypercube"),
            "M10d_Cubic": (70, 71, "70D hypercube"),
            "M11_Moebius": (2, 3, "Möebius band 10-times twisted"),
            "M12_Norm": (20, 20, "Isotropic multivariate Gaussian"),
            "M13a_Scurve": (2, 3, "2D S-curve"),
            "M13b_Spiral": (1, 13, "1D helix curve"),
            "Mbeta": (
                10,
                40,
                "Manifold generated with a smooth nonuniform pdf (see paper for description)",
            ),
            "Mn1_Nonlinear": (
                18,
                72,
                "Nonlinearly embedded manifold of high ID (see paper for description)",
            ),
            "Mn2_Nonlinear": (
                24,
                96,
                "Nonlinearly embedded manifold of high ID (see paper for description)",
            ),
            "Mp1_Paraboloid": (
                3,
                12,
                "3D paraboloid, nonlinearly embedded in (3(3+1))D space, according to a multivariate Burr distribution (alpha=1)",
            ),
            "Mp2_Paraboloid": (
                6,
                21,
                "6D paraboloid, nonlinearly embedded in (3*(6+1))D space, according to a multivariate Burr distribution (alpha=1)",
            ),
            "Mp3_Paraboloid": (
                9,
                30,
                "9D paraboloid, nonlinearly embedded in (3*(9+1))D space, according to a multivariate Burr distribution (alpha=1)",
            ),
        }
        import pandas as pd  # optional: only BenchmarkManifolds needs it

        self.truth = pd.DataFrame(
            self._dict_truth,
            index=["Intrinsic Dimension", "Number of variables", "Description"],
        ).T

        self.dict_gen = {
            # synthetic data
            "M1_Sphere": self._gen_sphere_data,
            "M2_Affine_3to5": self._gen_affine3_5_data,
            "M3_Nonlinear_4to6": self._gen_nonlinear4_6_data,
            "M4_Nonlinear": self._gen_nonlinear_data,
            "M5a_Helix1d": self._gen_helix1_data,
            "M5b_Helix2d": self._gen_helix2_data,
            "M6_Nonlinear": self._gen_nonlinear_data,
            "M7_Roll": self._gen_roll_data,
            "M8_Nonlinear": self._gen_nonlinear_data,
            "M9_Affine": self._gen_affine_data,
            "M10a_Cubic": self._gen_cubic_data,
            "M10b_Cubic": self._gen_cubic_data,
            "M10c_Cubic": self._gen_cubic_data,
            "M10d_Cubic": self._gen_cubic_data,
            "M11_Moebius": self._gen_moebius_data,
            "M12_Norm": self._gen_norm_data,
            "M13a_Scurve": self._gen_scurve_data,
            "M13b_Spiral": self._gen_spiral_data,
            "Mbeta": self._gen_campadelli_beta_data,
            "Mn1_Nonlinear": self._gen_campadelli_n_data,
            "Mn2_Nonlinear": self._gen_campadelli_n_data,
            "Mp1_Paraboloid": self._gen_paraboloid_data,
            "Mp2_Paraboloid": self._gen_paraboloid_data,
            "Mp3_Paraboloid": self._gen_paraboloid_data,
        }

    def generate(
        self,
        name: str = "all",
        n: int = 2500,
        dim: int = None,
        d: int = None,
        noise: float = 0.0,
    ):
        """
        Generates all datasets. A ground truth dict of intrinsic dimension and embedding dimension is in BenchmarkManifolds.dict_truth.keys()

        Parameters
        ----------
        n: int
            The number of sample points
        dim: int
            If generating a single dataset, choose the embedding dimension. Note that some datasets have restrictions on the chosen embedding dimension
        d: int
            If generating a single dataset, choose the intrinsic dimension. Note that some datasets have restrictions on the chosen intrinsic dimension
        noise: float, optional(default=0.0)
            The value of noise in data

        Returns
        -------
        data: a dict of np.arrays or a single np.array with shape (n, dim)
           Generated data
        """

        if self.noise_type == "normal":
            self.gen_noise = (
                lambda n, dim, noise: (self.random_state.randn(n, dim)) * noise
            )
        if self.noise_type == "uniform":
            self.gen_noise = (
                lambda n, dim, noise: (self.random_state.rand(n, dim) - 0.5) * noise
            )

        dict_data = {}
        if name == "all":
            for k, (d, dim, _) in self._dict_truth.items():
                data = self.dict_gen[k](n=n, dim=dim, d=d)
                dict_data[k] = data + self.gen_noise(n, dim, noise)
            return dict_data

        elif name in self._dict_truth.keys():
            if dim is None and d is None:
                data = self.dict_gen[name](n=n)
            elif dim is None:
                data = self.dict_gen[name](n=n, d=d)
            elif d is None:
                data = self.dict_gen[name](n=n, dim=dim)

            data = self.dict_gen[name](n=n, dim=dim, d=d)
            return data + self.gen_noise(n, dim, noise)

    def _gen_spiral_data(self, n, dim=3, d=1):
        assert d < dim
        assert d == 1
        assert dim >= 3

        t = 10 * np.pi * self.random_state.rand(n)
        data = np.vstack(
            [100 * np.cos(t), 100 * np.sin(t), t, np.zeros((dim - 3, n))]
        ).T
        assert data.shape == (n, dim)
        return data

    def _gen_helix1_data(self, n, dim=3, d=1):
        assert d < dim
        assert d == 1
        assert dim >= 3

        t = 2 * np.pi / n + self.random_state.rand(n) * 2 * np.pi
        data = np.vstack(
            [
                (2 + np.cos(8 * t)) * np.cos(t),
                (2 + np.cos(8 * t)) * np.sin(t),
                np.sin(8 * t),
                np.zeros((dim - 3, n)),
            ]
        ).T
        assert data.shape == (n, dim)
        return data

    def _gen_helix2_data(self, n, dim=3, d=2):
        assert d < dim
        assert d == 2
        assert dim >= 3
        r = 10 * np.pi * self.random_state.rand(n)
        p = 10 * np.pi * self.random_state.rand(n)
        data = np.vstack(
            [r * np.cos(p), r * np.sin(p), 0.5 * p, np.zeros((dim - 3, n))]
        ).T
        assert data.shape == (n, dim)
        return data

    def _gen_helicoid_data(self, n, dim=3, d=2):
        assert d <= dim
        assert d == 2
        assert dim >= 3

        u = 2 * np.pi / n + self.random_state.rand(n) * 2 * np.pi
        v = 5 * np.pi * self.random_state.rand(n)
        data = np.vstack(
            [np.cos(v), np.sin(v) * np.cos(v), u, np.zeros((dim - 3, n))]
        ).T
        assert data.shape == (n, dim)
        return data

    def _gen_roll_data(self, n, dim=3, d=2):
        assert d < dim
        assert dim >= 3
        assert d == 2

        t = 1.5 * np.pi * (1 + 2 * self.random_state.rand(n))
        p = 21 * self.random_state.rand(n)

        data = np.vstack(
            [t * np.cos(t), p, t * np.sin(t), np.zeros((dim - d - 1, n))]
        ).T
        assert data.shape == (n, dim)
        return data

    def _gen_scurve_data(self, n, dim=3, d=2):
        assert d < dim
        assert dim >= 3
        assert d == 2

        t = 3 * np.pi * (self.random_state.rand(n) - 0.5)
        p = 2.0 * self.random_state.rand(n)

        data = np.vstack(
            [np.sin(t), p, np.sign(t) * (np.cos(t) - 1), np.zeros((dim - d - 1, n)),]
        ).T
        assert data.shape == (n, dim)
        return data

    def _gen_sphere_data(self, n, dim, d):
        assert d < dim

        V = self.random_state.randn(n, d + 1)
        data = np.hstack(
            [V / np.sqrt((V ** 2).sum(axis=1))[:, None], np.zeros((n, dim - d - 1)),]
        )
        assert data.shape == (n, dim)
        return data

    def _gen_norm_data(self, n, dim, d):
        assert d <= dim

        norm_xyz = self.random_state.multivariate_normal(np.zeros(d), np.identity(d), n)
        data = np.hstack([norm_xyz, np.zeros((n, dim - d))])
        assert data.shape == (n, dim)
        return data

    def _gen_uniform_data(self, n, dim, d):
        assert d <= dim
        uniform_xyz = self.random_state.uniform(size=(n, d))
        data = np.hstack([uniform_xyz, np.zeros((n, dim - d))])
        assert data.shape == (n, dim)
        return data

    def _gen_cubic_data(self, n, dim, d):
        assert d < dim
        cubic_data = np.array([[]] * (d + 1))
        for i in range(d + 1):
            n_once = int(n / (2 * (d + 1)) + 1)
            # 1st side
            data_once = self.random_state.rand(d + 1, n_once)
            data_once[i] = 0
            cubic_data = np.hstack([cubic_data, data_once])
            # 2nd side
            data_once = self.random_state.rand(d + 1, n_once)
            data_once[i] = 1
            cubic_data = np.hstack([cubic_data, data_once])
        cubic_data = cubic_data.T[:n]
        data = np.hstack([cubic_data, np.zeros((n, dim - d - 1))])
        assert data.shape == (n, dim)
        return data

    def _gen_moebius_data(self, n, dim=3, d=2):
        assert dim == 3
        assert d == 2

        phi = self.random_state.rand(n) * 2 * np.pi
        rad = self.random_state.rand(n) * 2 - 1
        data = np.vstack(
            [
                (1 + 0.5 * rad * np.cos(5.0 * phi)) * np.cos(phi),
                (1 + 0.5 * rad * np.cos(5.0 * phi)) * np.sin(phi),
                0.5 * rad * np.sin(5.0 * phi),
            ]
        ).T

        assert data.shape == (n, dim)
        return data

    def _gen_affine_data(self, n, dim, d):
        assert dim >= d

        p = self.random_state.rand(d, n) * 5 - 2.5
        v = np.eye(dim, d)
        #     v = np.random.randint(0, 10, (dim, d))
        data = v.dot(p).T

        assert data.shape == (n, dim)
        return data

    def _gen_affine3_5_data(self, n, dim=5, d=3):
        assert dim == 5
        assert d == 3

        p = 4 * self.random_state.rand(d, n)
        A = np.array(
            [
                [1.2, -0.5, 0],
                [0.5, 0.9, 0],
                [-0.5, -0.2, 1],
                [0.4, -0.9, -0.1],
                [1.1, -0.3, 0],
            ]
        )
        b = np.array([[3, -1, 0, 0, 8]]).T
        data = A.dot(p) + b
        data = data.T

        assert data.shape == (n, dim)
        return data

    def _gen_nonlinear4_6_data(self, n, dim=6, d=4):
        assert dim == 6
        assert d == 4

        p0, p1, p2, p3 = self.random_state.rand(d, n)
        data = np.vstack(
            [
                p1 ** 2 * np.cos(2 * np.pi * p0),
                p2 ** 2 * np.sin(2 * np.pi * p0),
                p1 + p2 + (p1 - p3) ** 2,
                p1 - 2 * p2 + (p0 - p3) ** 2,
                -p1 - 2 * p2 + (p2 - p3) ** 2,
                p0 ** 2 - p1 ** 2 + p2 ** 2 - p3 ** 2,
            ]
        ).T

        assert data.shape == (n, dim)
        return data

    def _gen_nonlinear_data(self, n, dim, d):
        assert dim >= d
        m = int(dim / (2 * d))
        assert dim == 2 * m * d

        p = self.random_state.rand(d, n)
        F = np.zeros((2 * d, n))
        F[0::2, :] = np.cos(2 * np.pi * p)
        F[1::2, :] = np.sin(2 * np.pi * p)
        R = np.zeros((2 * d, n))
        R[0::2, :] = np.vstack([p[1:], p[0]])
        R[1::2, :] = np.vstack([p[1:], p[0]])
        D = (R * F).T
        data = np.hstack([D] * m)

        assert data.shape == (n, dim)
        return data

    def _gen_paraboloid_data(self, n, dim, d):
        assert dim == 3 * (d + 1)

        E = self.random_state.exponential(1, (d + 1, n))
        X = ((1 + E[1:] / E[0]) ** -1).T
        X = np.hstack([X, (X ** 2).sum(axis=1)[:, np.newaxis]])
        data = np.hstack([X, np.sin(X), X ** 2])

        assert data.shape == (n, dim)
        return data

    def _gen_campadelli_n_data(self, n, dim, d):
        assert dim == d * 4

        # Generate points drawn from a uniform distribution
        X = self.random_state.random(size=(n, d))
        temp1 = np.zeros((n, d))
        temp2 = np.zeros((n, d))

        # Extend the embeddingDity:
        for i in range(d):
            temp1[:, i] = np.tan(X[:, i] * np.cos(X[:, d - 1 - i]))
            temp2[:, i] = np.arctan(X[:, d - 1 - i] * np.sin(X[:, i]))

        # Create the final dataset:
        data = np.concatenate([temp1, temp2, temp1, temp2], axis=1)
        return data

    def _gen_campadelli_beta_data(self, n, dim, d, alpha=10, beta=0.5):
        assert dim == d * 4

        # Function to generate point drawn from a beta distribution
        X = self.random_state.beta(a=alpha, b=beta, size=((n, d)))

        temp1 = np.zeros((n, d))
        temp2 = np.zeros((n, d))

        # Extend the embeddingDity:
        for i in range(d):
            temp1[:, i] = X[:, i] * np.sin(np.cos(2 * np.pi * (X[:, i])))
            temp2[:, i] = X[:, i] * np.cos(np.sin(2 * np.pi * (X[:, i])))

        # Create the final dataset:
        data = np.concatenate([temp1, temp2, temp1, temp2], axis=1)
        return data


class SurfaceOfRevolution:
    """
    Generates a uniform sample from the surface obtained by rotating the graph of y = func(x),
    x in [xmin, xmax], around the x-axis (how="x") or the y-axis (how="y"), by rejection sampling
    on the area element tabulated on a grid of 20001 points.
    Adapted from the kmyim/scikit-dimension fork (John Harvey, Ka Man Yim) [IDkmyim]_.

    Parameters
    ----------
    func: callable
        Vectorised real-valued function defining the profile curve y = func(x)
    how: str, 'x' or 'y'
        Axis of rotation
    xmin: float
        Left boundary of the profile curve
    xmax: float
        Right boundary of the profile curve
    random_state: int, np.random.RandomState instance
        Random number generator
    """

    def __init__(
        self, func=lambda x: np.sin(x * np.pi), how="x", xmin=-1.0, xmax=1.0, random_state=None
    ):
        if not xmin < xmax:
            raise ValueError("The variable xmin must be less than xmax")
        if how not in ["x", "y"]:
            raise ValueError("The variable how must be 'x' or 'y'.")
        self.func = func
        self.how = how
        self.xmin = xmin
        self.xmax = xmax
        self.random_state = check_random_state(random_state)

        self.xvals = np.linspace(xmin, xmax, 20001)
        self.fvals = np.asarray(func(self.xvals), dtype=float)
        # area element per dx dtheta: |rotation radius| * sqrt(1 + f'(x)^2)
        radius = self.fvals if how == "x" else self.xvals
        self.jacobian_vals = np.abs(radius) * np.sqrt(1 + np.gradient(self.fvals, self.xvals) ** 2)

    def generate(self, n=10000):
        """
        Generates n points from the surface of revolution.

        Parameters
        ----------
        n: int
            Number of data points.

        Returns
        -------
        data: np.array, (npoints x 3)
            Generated data
        """
        rs = self.random_state
        jac_max = np.max(self.jacobian_vals)
        x = np.empty(0)
        while x.shape[0] < n:
            u = rs.uniform(self.xmin, self.xmax, n)
            jac = np.interp(u, self.xvals, self.jacobian_vals)
            x = np.concatenate((x, u[rs.uniform(0, jac_max, n) < jac]))
        x = x[:n]
        theta = rs.uniform(0, 2 * np.pi, n)
        f = np.interp(x, self.xvals, self.fvals)
        if self.how == "x":
            return np.stack((x, f * np.cos(theta), f * np.sin(theta)), axis=1)
        return np.stack((x * np.cos(theta), f, x * np.sin(theta)), axis=1)
