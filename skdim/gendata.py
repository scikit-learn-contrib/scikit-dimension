#
# BSD 3-Clause License
#
# Copyright (c) 2020, Jonathan Bac
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import numpy as np
from sklearn.utils.validation import check_random_state
from ._commonfuncs import hyperBall


def hyperSphere(n_points, n_dim, center=[], random_state=None):
    """
    Generates a sample from a uniform distribution on an hypersphere surface
    """
    random_state_ = check_random_state(random_state)
    vec = random_state_.randn(n_points, n_dim)
    vec /= np.linalg.norm(vec, axis=1)[:, None]
    return vec


def hyperTwinPeaks(n_points, n_dim=2, height=1, random_state=None):
    """ 
    Translated from Kerstin Johnsson's R package intrinsicDimension
    """
    random_state_ = check_random_state(random_state)
    base_coord = random_state_.uniform(size=(n_points, n_dim))
    _height = height * np.prod(np.sin(2 * np.pi * base_coord), axis=1, keepdims=1)
    return np.hstack((base_coord, _height))


def swissRoll3Sph(Ns, Nsph, a=1, b=2, nturn=1.5, h=4, random_state=None):
    """
    Generates a sample from a uniform distribution on a Swiss roll-surface, 
    possibly together with a sample from a uniform distribution on a 3-sphere
    inside the Swiss roll. Translated from Kerstin Johnsson's R package intrinsicDimension

    Parameters
    ----------

    Ns : int 
        Number of data points on the Swiss roll.

    Nsph : int
        Number of data points on the 3-sphere.

    a : int or float, default=1
        Minimal radius of Swiss roll and radius of 3-sphere.

    b : int or float, default=2
        Maximal radius of Swiss roll.

    nturn : int or float, default=1.5
        Number of turns of the surface. 

    h : int or float, default=4
        Height of Swiss roll.

    Returns
    -------
    
    np.array, (npoints x ndim)
    """
    random_state_ = check_random_state(random_state)

    if Ns > 0:
        omega = 2 * np.pi * nturn
        dl = lambda r: np.sqrt(b ** 2 + omega ** 2 * (a + b * r) ** 2)
        ok = np.zeros(1)
        while sum(ok) < Ns:
            r_samp = random_state_.uniform(size=3 * Ns)
            ok = random_state_.uniform(size=3 * Ns) < dl(r_samp) / dl(1)

        r_samp = r_samp[ok][:Ns]
        x = (a + b * r_samp) * np.cos(omega * r_samp)
        y = (a + b * r_samp) * np.sin(omega * r_samp)
        z = random_state_.uniform(-h, h, size=Ns)
        w = np.zeros(Ns)

    else:
        x = y = z = w = np.array([])

    if Nsph > 0:
        sph = hyperSphere(Nsph, 4, random_state=random_state_) * a
        x = np.concatenate((x, sph[:, 0]))
        y = np.concatenate((y, sph[:, 1]))
        z = np.concatenate((z, sph[:, 2]))
        w = np.concatenate((w, sph[:, 3]))

    return np.hstack((x[:, None], y[:, None], z[:, None], w[:, None]))


def lineDiskBall(n_points, random_state=None):
    """ 
    Generates a sample from a uniform distribution on a line, an oblong disk and an oblong ball
    Translated from ldbl function in Hideitsu Hino's package
    """
    random_state_ = check_random_state(random_state)

    line = np.hstack(
        (
            np.repeat(0, 5 * n_points)[:, None],
            np.repeat(0, 5 * n_points)[:, None],
            random_state_.uniform(-0.5, 0, size=5 * n_points)[:, None],
        )
    )
    disc = np.hstack(
        (
            random_state_.uniform(-1, 1, (13 * n_points, 2)),
            np.zeros(13 * n_points)[:, None],
        )
    )
    disc = disc[~(np.sqrt(np.sum(disc ** 2, axis=1)) > 1), :]
    disc = disc[:, [0, 2, 1]]
    disc[:, 2] = disc[:, 2] - min(disc[:, 2]) + max(line[:, 2])

    fb = random_state_.uniform(-0.5, 0.5, size=(n_points * 100, 3))
    rmID = np.where(np.sqrt(np.sum(fb ** 2, axis=1)) > 0.5)[0]

    if len(rmID) > 0:
        fb = fb[~(np.sqrt(np.sum(fb ** 2, axis=1)) > 0.5), :]

    fb = np.hstack((fb[:, :2], fb[:, [2]] + 0.5))
    fb[:, 2] = fb[:, 2] - min(fb[:, 2]) + max(disc[:, 2])

    #     if _sorted:
    #         fb = fb[order(fb[:, 2]),:]

    line2 = np.hstack(
        (
            np.repeat(0, 5 * n_points)[:, None],
            np.repeat(0, 5 * n_points)[:, None],
            random_state_.uniform(-0.5, 0, size=5 * n_points)[:, None],
        )
    )
    line2[:, 2] = line2[:, 2] - min(line2[:, 2]) + max(fb[:, 2])
    lineID = np.repeat(1, len(line))
    discID = np.repeat(2, len(disc))
    fbID = np.repeat(3, len(fb))
    line2ID = np.repeat(1, len(line2))
    x = np.vstack((line, disc, fb, line2))
    useID = np.sort(random_state_.choice(len(x), n_points, replace=False))
    x = x[useID, :]

    return x, np.concatenate((lineID, discID, fbID, line2ID), axis=0)[useID]


### Hein manifolds


class DataGenerator:
    # modified from https://github.com/stat-ml/GeoMLE
    # Original licence citation:
    # MIT License
    #
    # Copyright (c) 2019 Mokrov Nikita, Marina Gomtsyan, Maxim Panov and Yury Yanovich
    #
    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to the following conditions:
    #
    # The above copyright notice and this permission notice shall be included in all
    # copies or substantial portions of the Software.
    #
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    # SOFTWARE.

    def __init__(self, random_state: int = None, type_noise: str = "norm"):

        self.set_rng(random_state)
        self.set_gen_noise(type_noise)
        self.dict_gen = {
            # synthetic data
            "Helix1d": gen_helix1_data,
            "Helix2d": gen_helix2_data,
            "Helicoid": gen_helicoid_data,
            "Spiral": gen_spiral_data,
            "Roll": gen_roll_data,
            "Scurve": gen_scurve_data,
            "Star": gen_star_data,
            "Moebius": gen_moebius_data,
            "Sphere": gen_sphere_data,
            "Norm": gen_norm_data,
            "Uniform": gen_uniform_data,
            "Cubic": gen_cubic_data,
            "Affine_3to5": gen_affine3_5_data,
            "Affine": gen_affine_data,
            "Nonlinear_4to6": gen_nonlinear4_6_data,
            "Nonlinear": gen_nonlinear_data,
            "Paraboloid": gen_paraboloid_data,
        }

    def set_rng(self, random_state: int = None):
        if random_state is not None:
            np.random.seed(random_state)

    def set_gen_noise(self, type_noise: str):
        if not hasattr(self, "rng"):
            self.set_rng()
        if type_noise == "norm":
            self.gen_noise = np.random.randn
        if type_noise == "uniform":
            self.gen_noise = lambda n, dim: np.random.rand(n, dim) - 0.5

    def gen_data(
        self,
        name: str,
        n: int,
        dim: int,
        d: int,
        type_sample: str = "uniform",
        noise: float = 0.0,
    ):
        # Parameters:
        # --------------------
        # name: string
        #     Type of generetic data
        # n: int
        #     The number of sample points
        # dim: int
        #     The dimension of point
        # d: int
        #     The hyperplane dimension
        # noise: float, optional(default=0.0)
        #     The value of noise in data

        # Returns:
        # data: pd.Dataframe of shape (n, dim)
        #     The points
        assert name in self.dict_gen.keys(), "Name of data is unknown"
        if type_sample == "uniform":
            if name == "Sphere":
                sampler = np.random.randn
            else:
                sampler = np.random.rand
        elif type_sample == "nonuniform":
            if name == "Sphere":
                sampler = uniform_sampler
            else:
                sampler = bound_nonuniform_sampler
        else:
            assert False, "Check type_sample"

        data = self.dict_gen[name](n=n, dim=dim, d=d, sampler=sampler)
        noise = self.gen_noise(n, dim) * noise

        return data + noise


def bound_nonuniform_sampler(*args):
    x = np.random.randn(*args) * 0.1 + 0.5
    x[x < 0] = -x[x < 0]
    x[x > 1] = x[x > 1] - 1
    x[x < 0] = -x[x < 0]
    return x


def uniform_sampler(*args):
    x = np.random.rand(*args)
    x = (x - 0.5) * 3
    return x


def gen_spiral_data(n, dim, d, sampler):
    assert d < dim
    assert d == 1
    assert dim >= 3
    t = 10 * np.pi * sampler(n)
    data = np.vstack([100 * np.cos(t), 100 * np.sin(t), t, np.zeros((dim - 3, n))]).T
    assert data.shape == (n, dim)
    return data


def gen_helix1_data(n, dim, d, sampler):
    assert d < dim
    assert d == 1
    assert dim >= 3
    t = 2 * np.pi / n + sampler(n) * 2 * np.pi
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


def gen_helix2_data(n, dim, d, sampler):
    assert d < dim
    assert d == 2
    assert dim >= 3
    r = 10 * np.pi * sampler(n)
    p = 10 * np.pi * sampler(n)
    data = np.vstack([r * np.cos(p), r * np.sin(p), 0.5 * p, np.zeros((dim - 3, n))]).T
    assert data.shape == (n, dim)
    return data


def gen_helicoid_data(n, dim, d, sampler):
    assert d <= dim
    assert d == 2
    assert dim >= 3
    u = 2 * np.pi / n + sampler(n) * 2 * np.pi
    v = 5 * np.pi * sampler(n)
    data = np.vstack([np.cos(v), np.sin(v) * np.cos(v), u, np.zeros((dim - 3, n))]).T
    assert data.shape == (n, dim)
    return data


def gen_roll_data(n, dim, d, sampler):
    assert d < dim
    assert dim >= 3
    assert d == 2
    t = 1.5 * np.pi * (1 + 2 * sampler(n))
    p = 21 * sampler(n)

    data = np.vstack([t * np.cos(t), p, t * np.sin(t), np.zeros((dim - d - 1, n))]).T
    assert data.shape == (n, dim)
    return data


def gen_scurve_data(n, dim, d, sampler):
    assert d < dim
    assert dim >= 3
    assert d == 2
    t = 3 * np.pi * (sampler(n) - 0.5)
    p = 2.0 * sampler(n)

    data = np.vstack(
        [np.sin(t), p, np.sign(t) * (np.cos(t) - 1), np.zeros((dim - d - 1, n))]
    ).T
    assert data.shape == (n, dim)
    return data


def gen_sphere_data(n, dim, d, sampler):
    assert d < dim
    #     V = np.random.randn(n, d + 1)
    V = sampler(n, d + 1)
    data = np.hstack(
        [V / np.sqrt((V ** 2).sum(axis=1))[:, None], np.zeros((n, dim - d - 1))]
    )
    assert data.shape == (n, dim)
    return data


def gen_norm_data(n, dim, d, sampler):
    assert d <= dim
    norm_xyz = np.random.multivariate_normal(np.zeros(d), np.identity(d), n)
    data = np.hstack([norm_xyz, np.zeros((n, dim - d))])
    assert data.shape == (n, dim)
    return data


def gen_uniform_data(n, dim, d, sampler):
    assert d <= dim
    uniform_xyz = np.random.uniform(size=(n, d))
    data = np.hstack([uniform_xyz, np.zeros((n, dim - d))])
    assert data.shape == (n, dim)
    return data


def gen_cubic_data(n, dim, d, sampler):
    assert d < dim
    cubic_data = np.array([[]] * (d + 1))
    for i in range(d + 1):
        n_once = int(n / (2 * (d + 1)) + 1)
        # 1st side
        data_once = sampler(d + 1, n_once)
        data_once[i] = 0
        cubic_data = np.hstack([cubic_data, data_once])
        # 2nd side
        data_once = sampler(d + 1, n_once)
        data_once[i] = 1
        cubic_data = np.hstack([cubic_data, data_once])
    cubic_data = cubic_data.T[:n]
    data = np.hstack([cubic_data, np.zeros((n, dim - d - 1))])
    assert data.shape == (n, dim)
    return data


def gen_moebius_data(n, dim, d, sampler):
    assert dim == 3
    assert d == 2

    phi = sampler(n) * 2 * np.pi
    rad = sampler(n) * 2 - 1
    data = np.vstack(
        [
            (1 + 0.5 * rad * np.cos(5.0 * phi)) * np.cos(phi),
            (1 + 0.5 * rad * np.cos(5.0 * phi)) * np.sin(phi),
            0.5 * rad * np.sin(5.0 * phi),
        ]
    ).T

    assert data.shape == (n, dim)
    return data


def gen_affine_data(n, dim, d, sampler):
    assert dim >= d

    p = sampler(d, n) * 5 - 2.5
    v = np.eye(dim, d)
    #     v = np.random.randint(0, 10, (dim, d))
    data = v.dot(p).T

    assert data.shape == (n, dim)
    return data


def gen_affine3_5_data(n, dim, d, sampler):
    assert dim == 5
    assert d == 3

    p = 4 * sampler(d, n)
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


def gen_nonlinear4_6_data(n, dim, d, sampler):
    assert dim == 6
    assert d == 4

    p0, p1, p2, p3 = sampler(d, n)
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


def gen_nonlinear_data(n, dim, d, sampler):
    assert dim >= d
    m = int(dim / (2 * d))
    assert dim == 2 * m * d

    p = sampler(d, n)
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


def gen_paraboloid_data(n, dim, d, sampler):
    assert dim == 3 * (d + 1)

    E = np.random.exponential(1, (d + 1, n))
    X = ((1 + E[1:] / E[0]) ** -1).T
    X = np.hstack([X, (X ** 2).sum(axis=1)[:, np.newaxis]])
    data = np.hstack([X, np.sin(X), X ** 2])

    assert data.shape == (n, dim)
    return data


def gen_star_data(n, dim, d, sampler):
    assert dim >= d
    assert d == 1
    assert dim >= 2

    t = np.pi - sampler(n) * 2 * np.pi
    omega = 5
    data = np.concatenate(
        (
            ((1 + 0.3 * np.cos(omega * t)) * np.cos(t)).reshape(-1, 1),
            ((1 + 0.3 * np.cos(omega * t)) * np.sin(t)).reshape(-1, 1),
            np.zeros((n, dim - 2)),
        ),
        axis=1,
    )

    assert data.shape == (n, dim)
    return data

