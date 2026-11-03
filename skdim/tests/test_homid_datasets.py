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


def test_stiefel_matrix_preserves_raw_qr_sign_convention():
    # Preserve the archived LAPACK QR convention instead of applying a
    # column-sign correction. Its first frame coordinate is nonpositive.
    points = ds.stiefel_matrix(N, 5, 1, random_state=42)
    assert np.all(points[:, 0] <= 0)
    assert np.all(points[:, 0] < -1e-10)


@pytest.mark.parametrize("p,k", [(5, 2), (5, 4), (2, 1)])
def test_stiefel_vector_orthogonal_factor(p, k):
    points = ds.stiefel_vector(N, p, k, random_state=42)
    split = comb(p, k)
    plane = points[:, :split]
    rotations = points[:, split:].reshape(N, k, k)
    np.testing.assert_allclose(np.sum(plane**2, axis=1), 1, atol=ATOL)
    gram = rotations.swapaxes(-1, -2) @ rotations
    np.testing.assert_allclose(
        gram, np.broadcast_to(np.eye(k), gram.shape), atol=ATOL
    )
    determinants = np.linalg.det(rotations)
    np.testing.assert_allclose(np.abs(determinants), 1, atol=ATOL)
    # O(k), including O(1), retains both determinant components.
    assert 20 <= np.count_nonzero(determinants > 0) <= 80
    np.testing.assert_allclose(np.sum(points**2, axis=1), 1 + k, atol=ATOL)
    if k == 1:
        np.testing.assert_array_equal(np.abs(rotations), np.ones((N, 1, 1)))


@pytest.mark.parametrize(
    "p,k1,k2",
    [
        (5, 2, 2),
        (5, 3, 2),
        (5, 2, 3),
        (5, 4, 3),
        (4, 2, 2),
        (2, 1, 1),
        (5, 5, 2),
        (5, 2, 5),
        (3, 3, 3),
    ],
)
def test_flag_factors_are_nested_subspaces(p, k1, k2):
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
    smaller = first if k1 <= k2 else second
    np.testing.assert_allclose(first @ second, smaller, atol=ATOL)
    # The two prefix ranks may sum to more than p. Full rank is also valid.
    if max(k1, k2) == p:
        larger = first if k1 >= k2 else second
        np.testing.assert_allclose(
            larger, np.broadcast_to(np.eye(p), larger.shape), atol=ATOL
        )


@pytest.mark.parametrize("p,k", [(4, 2), (5, 2), (3, 3)])
def test_equal_rank_flag_is_a_self_tensor(p, k):
    tensors = ds.flag_vector(N, p, k, k, random_state=42).reshape(
        N, comb(p, k), comb(p, k)
    )
    np.testing.assert_allclose(tensors, tensors.swapaxes(-1, -2), atol=ATOL)
    np.testing.assert_allclose(tensors @ tensors, tensors, atol=ATOL)
    np.testing.assert_allclose(
        np.trace(tensors, axis1=1, axis2=2), 1, atol=ATOL
    )
    assert np.all(np.diagonal(tensors, axis1=1, axis2=2) >= -ATOL)


def test_full_rank_equal_flag_is_constant():
    # Both minors are det(Q), so their product is identically one.
    points = ds.flag_vector(N, 5, 5, 5, random_state=42)
    np.testing.assert_allclose(points, np.ones((N, 1)), atol=ATOL)


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
def test_invalid_flag_prefix_rank(parameter, bad):
    parameters = dict(k1=2, k2=1)
    parameters[parameter] = bad
    with pytest.raises((TypeError, ValueError)):
        ds.flag_vector(1, 5, **parameters, random_state=42)


@pytest.mark.parametrize("k1,k2", [(6, 1), (1, 6), (6, 6)])
def test_flag_prefix_ranks_must_not_exceed_ambient_dimension(k1, k2):
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


def _serial_minor_vector(frame):
    p, k = frame.shape
    return np.array(
        [
            np.linalg.det(frame[list(rows), :])
            for rows in combinations(range(p), k)
        ]
    )


def _serial_qr_frame(gaussians, correct_signs=True):
    frame, triangular = np.linalg.qr(gaussians, mode="reduced")
    if correct_signs:
        for j in range(frame.shape[1]):
            if triangular[j, j] < 0:
                frame[:, j] *= -1
    return frame


def _serial_reference(name, n, p, random_state, **parameters):
    """Use scalar draws and determinants to check the archived formulas."""
    rng = np.random.RandomState(random_state)
    points = []
    for _ in range(n):
        if name == "flag_vector":
            k1, k2 = parameters["k1"], parameters["k2"]
            width = max(k1, k2)
        else:
            width = parameters["k"]
        frame = _serial_qr_frame(
            rng.normal(size=(p, width)),
            correct_signs=name != "stiefel_matrix",
        )
        if name == "grassmannian_projector":
            point = (frame @ frame.T).ravel()
        elif name == "grassmannian_vector":
            point = _serial_minor_vector(frame)
        elif name == "stiefel_matrix":
            point = frame.ravel()
        elif name == "stiefel_vector":
            rotation = _serial_qr_frame(rng.normal(size=(width, width)))
            point = np.concatenate(
                [_serial_minor_vector(frame), rotation.ravel()]
            )
        else:
            first = _serial_minor_vector(frame[:, :k1])
            second = _serial_minor_vector(frame[:, :k2])
            point = np.outer(first, second).ravel()
        points.append(point)
    return np.asarray(points)


@pytest.mark.parametrize(
    "name,kwargs,width",
    CASES
    + [
        pytest.param("stiefel_vector", {"p": 2, "k": 1}, 3, id="stiefel-O1"),
        pytest.param(
            "flag_vector",
            {"p": 5, "k1": 1, "k2": 3},
            50,
            id="flag-reversed-ranks",
        ),
        pytest.param(
            "flag_vector",
            {"p": 5, "k1": 5, "k2": 2},
            10,
            id="flag-full-prefix",
        ),
        pytest.param(
            "flag_vector",
            {"p": 5, "k1": 5, "k2": 5},
            1,
            id="flag-full-equal-prefixes",
        ),
    ],
)
def test_matches_independent_serial_reference(name, kwargs, width):
    expected = _serial_reference(name, N, random_state=42, **kwargs)
    actual = getattr(ds, name)(N, **kwargs, random_state=42)
    assert actual.shape == (N, width)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=ATOL)
