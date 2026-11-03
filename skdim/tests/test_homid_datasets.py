# SPDX-License-Identifier: BSD-3-Clause
"""Geometric and random-state contracts for the HomID dataset generators."""

from itertools import combinations
from math import comb
import warnings

import numpy as np
import pytest
from joblib import Memory

from skdim import datasets as ds


# One percent of the original HomID generator's 10,000-point sample count.
N = 100
ATOL = 2e-12

CASES = [
    pytest.param(
        "grassmannian_projector",
        {"p": 5, "k": 2},
        25,
        id="grassmann-projector",
    ),
    pytest.param(
        "grassmannian_vector", {"p": 5, "k": 2}, 10, id="grassmann-vector"
    ),
    pytest.param("stiefel_matrix", {"p": 5, "k": 2}, 10, id="stiefel-matrix"),
    pytest.param("stiefel_vector", {"p": 5, "k": 2}, 14, id="stiefel-vector"),
    pytest.param(
        "flag_vector", {"p": 5, "k1": 2, "k2": 1}, 50, id="flag-vector"
    ),
]
FRAME_GENERATORS = [
    "grassmannian_projector",
    "grassmannian_vector",
    "stiefel_matrix",
    "stiefel_vector",
]


def _assert_same_state(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def _projector_from_minors(vectors, p, k):
    """Recover a projector by contracting a unit decomposable k-vector."""
    coordinates = {rows: i for i, rows in enumerate(combinations(range(p), k))}
    contractions = np.zeros((len(vectors), p, comb(p, k - 1)))
    for j, rows in enumerate(combinations(range(p), k - 1)):
        for i in range(p):
            if i in rows:
                continue
            sign = (-1) ** sum(row < i for row in rows)
            index = coordinates[tuple(sorted((i,) + rows))]
            contractions[:, i, j] = sign * vectors[:, index]
    return contractions @ contractions.swapaxes(-1, -2)


@pytest.mark.parametrize("name,kwargs,width", CASES)
def test_shape_finite_dtype_and_seeds(name, kwargs, width):
    generate = getattr(ds, name)
    first = generate(N, **kwargs, random_state=42)
    repeated = generate(N, **kwargs, random_state=42)
    changed = generate(N, **kwargs, random_state=43)
    assert first.shape == (N, width)
    assert first.dtype == np.float64
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, repeated)
    assert not np.allclose(first, changed)


@pytest.mark.parametrize("name,kwargs,width", CASES)
@pytest.mark.parametrize("n", [0, 1, np.int64(3)])
def test_empty_and_small_sample_counts(name, kwargs, width, n):
    result = getattr(ds, name)(n, **kwargs, random_state=42)
    assert result.shape == (n, width)
    assert result.dtype == np.float64
    assert np.isfinite(result).all()


@pytest.mark.parametrize("name,kwargs,width", CASES)
def test_integer_seed_preserves_numpy_global_state(name, kwargs, width):
    before = np.random.get_state()
    getattr(ds, name)(N, **kwargs, random_state=42)
    _assert_same_state(before, np.random.get_state())


@pytest.mark.parametrize("name,kwargs,width", CASES)
def test_random_state_instance_advances(name, kwargs, width):
    generate = getattr(ds, name)
    random_state = np.random.RandomState(42)
    before = random_state.get_state()
    first = generate(N, **kwargs, random_state=random_state)
    after = random_state.get_state()
    assert before[2:] != after[2:] or not np.array_equal(before[1], after[1])
    second = generate(N, **kwargs, random_state=random_state)
    assert not np.allclose(first, second)
    repeated = generate(N, **kwargs, random_state=np.random.RandomState(42))
    np.testing.assert_array_equal(first, repeated)


def test_grassmannian_projector_geometry():
    p, k = 5, 2
    points = ds.grassmannian_projector(N, p, k, random_state=42).reshape(
        N, p, p
    )
    np.testing.assert_allclose(points, points.swapaxes(-1, -2), atol=ATOL)
    np.testing.assert_allclose(points @ points, points, atol=ATOL)
    np.testing.assert_allclose(
        np.trace(points, axis1=1, axis2=2), k, atol=ATOL
    )
    np.testing.assert_allclose(np.sum(points**2, axis=(1, 2)), k, atol=ATOL)


def test_grassmannian_vector_plucker_coordinates():
    # Lexicographic coordinates are 12, 13, 14, 23, 24, 34.
    points = ds.grassmannian_vector(N, 4, 2, random_state=42)
    np.testing.assert_allclose(np.sum(points**2, axis=1), 1, atol=ATOL)
    relation = (
        points[:, 0] * points[:, 5]
        - points[:, 1] * points[:, 4]
        + points[:, 2] * points[:, 3]
    )
    np.testing.assert_allclose(relation, 0, atol=ATOL)
    projectors = _projector_from_minors(points, 4, 2)
    np.testing.assert_allclose(projectors @ projectors, projectors, atol=ATOL)
    np.testing.assert_allclose(
        np.trace(projectors, axis1=1, axis2=2), 2, atol=ATOL
    )
    # Absolute or canonicalized minor signs change the oriented-vector law.
    assert np.any(points[:, 0] > 0)
    assert np.any(points[:, 0] < 0)


def test_stiefel_matrix_geometry():
    p, k = 5, 2
    frames = ds.stiefel_matrix(N, p, k, random_state=42).reshape(N, p, k)
    gram = frames.swapaxes(-1, -2) @ frames
    np.testing.assert_allclose(
        gram, np.broadcast_to(np.eye(k), gram.shape), atol=ATOL
    )
    np.testing.assert_allclose(np.sum(frames**2, axis=(1, 2)), k, atol=ATOL)


def test_stiefel_sampling_has_no_raw_qr_hemisphere_bias():
    # A Haar frame's first column is uniform on the sphere. At N=100 the
    # 20%-80% sign bounds are broad (six binomial standard deviations).
    # Uncorrected LAPACK QR instead gives Q[0, 0] <= 0 at every point.
    points = ds.stiefel_matrix(N, 5, 1, random_state=42)
    positive_count = np.count_nonzero(points[:, 0] > 0)
    assert 20 <= positive_count <= 80
    np.testing.assert_allclose(np.mean(points, axis=0), 0, atol=0.2)


@pytest.mark.parametrize("p,k", [(5, 2), (5, 4), (2, 1)])
def test_stiefel_vector_special_orthogonal_factor(p, k):
    points = ds.stiefel_vector(N, p, k, random_state=42)
    split = comb(p, k)
    plane = points[:, :split]
    rotations = points[:, split:].reshape(N, k, k)
    np.testing.assert_allclose(np.sum(plane**2, axis=1), 1, atol=ATOL)
    gram = rotations.swapaxes(-1, -2) @ rotations
    np.testing.assert_allclose(
        gram, np.broadcast_to(np.eye(k), gram.shape), atol=ATOL
    )
    np.testing.assert_allclose(np.linalg.det(rotations), 1, atol=ATOL)
    np.testing.assert_allclose(np.sum(points**2, axis=1), 1 + k, atol=ATOL)
    if k == 1:
        np.testing.assert_array_equal(rotations, np.ones((N, 1, 1)))


@pytest.mark.parametrize(
    "p,k1,k2", [(5, 2, 2), (5, 2, 1), (4, 2, 2), (2, 1, 1)]
)
def test_flag_factors_are_disjoint_orthogonal_subspaces(p, k1, k2):
    points = ds.flag_vector(N, p, k1, k2, random_state=42)
    tensors = points.reshape(N, comb(p, k1), comb(p, k2))
    left, singular_values, right = np.linalg.svd(tensors, full_matrices=False)
    np.testing.assert_allclose(singular_values[:, 0], 1, atol=ATOL)
    np.testing.assert_allclose(singular_values[:, 1:], 0, atol=ATOL)
    first = _projector_from_minors(left[:, :, 0], p, k1)
    second = _projector_from_minors(right[:, 0, :], p, k2)
    np.testing.assert_allclose(first @ first, first, atol=ATOL)
    np.testing.assert_allclose(second @ second, second, atol=ATOL)
    np.testing.assert_allclose(
        np.trace(first, axis1=1, axis2=2), k1, atol=ATOL
    )
    np.testing.assert_allclose(
        np.trace(second, axis1=1, axis2=2), k2, atol=ATOL
    )
    # Reusing leading prefixes produces nested planes and fails this test.
    np.testing.assert_allclose(first @ second, 0, atol=ATOL)
    if k1 + k2 == p:
        np.testing.assert_allclose(
            first + second, np.broadcast_to(np.eye(p), first.shape), atol=ATOL
        )


def test_full_width_flag_uses_both_components_of_orthogonal_group():
    p, k1, k2 = 4, 2, 2
    tensors = ds.flag_vector(N, p, k1, k2, random_state=42).reshape(
        N, comb(p, k1), comb(p, k2)
    )
    rows = list(combinations(range(p), k1))
    columns = {row: i for i, row in enumerate(combinations(range(p), k2))}
    orientation = np.zeros(N)
    for i, first in enumerate(rows):
        second = tuple(j for j in range(p) if j not in first)
        inversions = sum(a > b for a in first for b in second)
        orientation += (-1) ** inversions * tensors[:, i, columns[second]]
    # Laplace expansion: this is det(Q), not a new independently sampled sign.
    np.testing.assert_allclose(np.abs(orientation), 1, atol=ATOL)
    assert 20 <= np.count_nonzero(orientation > 0) <= 80


@pytest.mark.parametrize("name", FRAME_GENERATORS)
@pytest.mark.parametrize("k", [1, 3])
def test_frame_width_boundaries(name, k):
    p = 4
    width = {
        "grassmannian_projector": p**2,
        "grassmannian_vector": comb(p, k),
        "stiefel_matrix": p * k,
        "stiefel_vector": comb(p, k) + k**2,
    }[name]
    result = getattr(ds, name)(1, np.int64(p), np.int64(k), random_state=42)
    assert result.shape == (1, width)
    assert result.dtype == np.float64
    assert np.isfinite(result).all()


@pytest.mark.parametrize("name,kwargs,width", CASES)
@pytest.mark.parametrize(
    "bad", [-1, True, np.bool_(True), 1.5, 2.0, "2", None]
)
def test_invalid_sample_count(name, kwargs, width, bad):
    with pytest.raises((TypeError, ValueError)):
        getattr(ds, name)(bad, **kwargs, random_state=42)


@pytest.mark.parametrize("name,kwargs,width", CASES)
@pytest.mark.parametrize(
    "bad", [-1, 0, 1, True, np.bool_(False), 2.5, 5.0, "5", None]
)
def test_invalid_ambient_dimension(name, kwargs, width, bad):
    parameters = dict(kwargs, p=bad)
    with pytest.raises((TypeError, ValueError)):
        getattr(ds, name)(1, **parameters, random_state=42)


@pytest.mark.parametrize("name", FRAME_GENERATORS)
@pytest.mark.parametrize(
    "bad", [-1, 0, 5, 6, True, np.bool_(True), 1.5, 2.0, "2", None]
)
def test_invalid_frame_width(name, bad):
    with pytest.raises((TypeError, ValueError)):
        getattr(ds, name)(1, 5, bad, random_state=42)


@pytest.mark.parametrize("parameter", ["k1", "k2"])
@pytest.mark.parametrize(
    "bad", [-1, 0, True, np.bool_(True), 1.5, 2.0, "2", None]
)
def test_invalid_flag_block_width(parameter, bad):
    parameters = dict(k1=2, k2=1)
    parameters[parameter] = bad
    with pytest.raises((TypeError, ValueError)):
        ds.flag_vector(1, 5, **parameters, random_state=42)


@pytest.mark.parametrize("k1,k2", [(3, 3), (5, 1), (1, 5)])
def test_flag_blocks_must_fit_ambient_space(k1, k2):
    with pytest.raises(ValueError):
        ds.flag_vector(1, 5, k1, k2, random_state=42)


@pytest.mark.parametrize("name,kwargs,width", CASES)
def test_seeded_public_generator_can_be_cached(name, kwargs, width, tmp_path):
    cached = Memory(location=tmp_path, verbose=0).cache(getattr(ds, name))
    assert not cached.check_call_in_cache(N, **kwargs, random_state=42)
    first = cached(N, **kwargs, random_state=42)
    assert cached.check_call_in_cache(N, **kwargs, random_state=42)
    np.testing.assert_array_equal(cached(N, **kwargs, random_state=42), first)
    assert not cached.check_call_in_cache(N, **kwargs, random_state=43)


@pytest.mark.parametrize("name,kwargs,width", CASES)
def test_seeded_sample_prefix_is_independent_of_requested_count(
    name, kwargs, width
):
    generate = getattr(ds, name)
    short = generate(N, **kwargs, random_state=42)
    longer = generate(N + 1, **kwargs, random_state=42)
    np.testing.assert_array_equal(short, longer[:N])


def test_large_feature_count_warns_before_generation():
    # The output is only about 0.5 MiB, despite its combinatorial width.
    with pytest.warns(UserWarning, match="Large HomID dataset"):
        points = ds.flag_vector(1, 10, 5, 5, random_state=42)
    assert points.shape == (1, comb(10, 5) ** 2)
    assert np.isfinite(points).all()


@pytest.mark.parametrize(
    "name,args",
    [
        ("grassmannian_vector", (1, 100, 50)),
        ("flag_vector", (1, 40, 20, 20)),
    ],
)
def test_unrepresentable_output_shape_is_rejected(name, args):
    # These dimensions exceed NumPy's index limit; no allocation is needed.
    with pytest.raises(ValueError):
        getattr(ds, name)(*args, random_state=42)


def test_empty_large_feature_dataset_does_not_warn_about_resources():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        points = ds.flag_vector(0, 10, 5, 5, random_state=42)
    assert points.shape == (0, comb(10, 5) ** 2)
    assert points.dtype == np.float64
    assert not [
        warning
        for warning in caught
        if issubclass(warning.category, UserWarning)
    ]


@pytest.mark.parametrize("name,kwargs,width", CASES)
def test_generation_creates_no_automatic_files(
    name, kwargs, width, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    points = getattr(ds, name)(N, **kwargs, random_state=42)
    assert points.shape == (N, width)
    assert list(tmp_path.iterdir()) == []
