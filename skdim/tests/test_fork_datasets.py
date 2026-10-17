# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest
from scipy.integrate import quad

from skdim import datasets as ds
from skdim import embeddings as emb
from skdim.id import TwoNN

# tolerances fixed before running: ~3-4 standard errors at the sample sizes used
N = 20000

GENERATORS = [
    (lambda rs: ds.hyperSphere(100, 4, radius=2.0, center=np.ones(5), random_state=rs), (100, 5)),
    (lambda rs: ds.hyperTwinPeaksUniform(100, d=3, random_state=rs), (100, 4)),
    (lambda rs: ds.toroidal_spiral(100, random_state=rs), (100, 3)),
    (lambda rs: ds.torus(100, random_state=rs), (100, 3)),
    (lambda rs: ds.dumbbell(100, random_state=rs), (100, 3)),
    (lambda rs: ds.SurfaceOfRevolution(random_state=rs).generate(100), (100, 3)),
    (
        lambda rs: ds.product(
            100, ds.hyperSphere, {"d": 1, "random_state": rs}, ds.hyperBall, {"d": 2, "random_state": rs + 1}
        ),
        (100, 4),
    ),
]


@pytest.mark.parametrize("gen,shape", GENERATORS)
def test_shape_and_seed(gen, shape):
    a, b, c = gen(0), gen(0), gen(1)
    assert a.shape == shape
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_hyperSphere_default_unchanged_and_kwargs():
    rs = np.random.RandomState(3)
    legacy = rs.randn(50, 4)
    legacy /= np.linalg.norm(legacy, axis=1)[:, None]
    np.testing.assert_array_equal(ds.hyperSphere(50, 3, random_state=3), legacy)
    c = np.array([1.0, -2.0, 0.5, 3.0])
    x = ds.hyperSphere(50, 3, radius=2.5, center=c, random_state=3)
    np.testing.assert_allclose(x, c + 2.5 * legacy, rtol=0, atol=1e-12)


def test_product_does_not_mutate_kwargs():
    kw = {"d": 2, "random_state": 0}
    ds.product(10, ds.hyperBall, kw, ds.hyperBall, kw)
    assert kw == {"d": 2, "random_state": 0}


def test_hyperTwinPeaksUniform_on_surface_and_uniform():
    h = 1.0
    x = ds.hyperTwinPeaksUniform(N, d=1, height=h, random_state=0)
    np.testing.assert_allclose(x[:, 1], h * np.sin(2 * np.pi * x[:, 0]), atol=1e-12)
    # arc-length density of x is proportional to sqrt(1 + (2 pi h cos(2 pi x))^2)
    w = lambda t: np.sqrt(1 + (2 * np.pi * h * np.cos(2 * np.pi * t)) ** 2)
    expected = quad(lambda t: np.cos(2 * np.pi * t) ** 2 * w(t), 0, 1)[0] / quad(w, 0, 1)[0]
    assert abs(np.mean(np.cos(2 * np.pi * x[:, 0]) ** 2) - expected) < 0.01


def test_torus_on_surface_and_uniform():
    r1, r2 = 2.0, 0.5
    x = ds.torus(N, r1=r1, r2=r2, random_state=0)
    rho = np.hypot(x[:, 0], x[:, 1])
    np.testing.assert_allclose((rho - r1) ** 2 + x[:, 2] ** 2, r2**2, atol=1e-12)
    # theta density (r1 + r2 cos theta) / (2 pi r1) gives E[cos theta] = r2 / (2 r1)
    assert abs(np.mean((rho - r1) / r2) - r2 / (2 * r1)) < 0.015


def test_toroidal_spiral_on_torus():
    r1, r2 = 2.0, 0.5
    x = ds.toroidal_spiral(1000, r1=r1, r2=r2, random_state=0)
    rho = np.hypot(x[:, 0], x[:, 1])
    np.testing.assert_allclose((rho - r1) ** 2 + x[:, 2] ** 2, r2**2, atol=1e-12)
    with pytest.raises(ValueError):
        ds.torus(10, r1=0.1, r2=0.2)


def test_surface_of_revolution():
    s = ds.SurfaceOfRevolution(random_state=0)
    x = s.generate(1000)
    np.testing.assert_allclose(np.hypot(x[:, 1], x[:, 2]), np.abs(np.sin(np.pi * x[:, 0])), atol=1e-6)
    # rotating y = x, x in [0, 1], around the y-axis: area element ~ x, so E[x] = 2/3
    y = ds.SurfaceOfRevolution(lambda t: t, how="y", xmin=0.0, xmax=1.0, random_state=0).generate(N)
    np.testing.assert_allclose(y[:, 1], np.hypot(y[:, 0], y[:, 2]), atol=1e-12)
    assert abs(np.mean(y[:, 1]) - 2 / 3) < 0.01


def test_lorenz_attractor():
    x = ds.lorenz_attractor(n=500, tmax=10.0)
    assert x.shape == (500, 3)
    np.testing.assert_array_equal(x, ds.lorenz_attractor(n=500, tmax=10.0))
    np.testing.assert_array_equal(x[0], [0, 1, 1.05])
    assert np.all(np.abs(x) < 60)


@pytest.mark.parametrize(
    "data,dim",
    [
        (ds.torus(2000, random_state=0), 2),
        (ds.dumbbell(2000, random_state=0), 2),
        (ds.toroidal_spiral(2000, random_state=0), 1),
        (ds.hyperTwinPeaksUniform(2000, d=2, random_state=0), 2),
    ],
)
def test_twonn_sanity(data, dim):
    assert abs(TwoNN().fit(data).dimension_ - dim) < 0.3


def test_monomials():
    np.testing.assert_array_equal(emb._generate_monomials(2, 2), [[2, 0], [1, 1], [0, 2]])
    assert emb._generate_monomials(3, 4).shape == (15, 3)  # binom(3 + 4 - 1, 4)
    np.testing.assert_array_equal(emb.monomial_embedding(np.array([[2.0, 3.0]]), 2), [[4, 6, 9]])
    np.testing.assert_array_equal(emb.monomial_embedding(np.ones((4, 2)), 0), np.ones((4, 1)))


def test_delay_embedding():
    x = np.arange(10)[:, None]
    y = emb.delay_embedding(x)
    assert y.shape == (8, 3)
    np.testing.assert_array_equal(y[0], [0, 1, 2])
    np.testing.assert_array_equal(emb.delay_embedding(x, 3), y)
    z = emb.delay_embedding(np.arange(20).reshape(10, 2), [1, 0])
    assert z.shape == (9, 4)
    np.testing.assert_array_equal(z[0], [0, 1, 2, 3])
    with pytest.raises(ValueError):
        emb.delay_embedding(x, 11)
