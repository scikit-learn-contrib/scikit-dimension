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
import pandas as pd
from sklearn.utils.validation import check_random_state
from sklearn.datasets import fetch_openml
from scipy.special import gammainc
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def product(n, sample_1, sample_1_kwargs, sample_2, sample_2_kwargs):
    """Create a sample from a product using two methods in this module.

    Parameters
    n   int
        number of points to sample
    sample_1    method
        function to create first sample
    sample_1_kwargs dict
        dictionary of arguments to pass to sample_1
    sample_2    method
        function to create first sample
    sample_2_kwargs dict
        dictionary of arguments to pass to sample_2
    """
    sample_1_kwargs.update({"n": n})
    sample_2_kwargs.update({"n": n})
    sample_1_points = sample_1(**sample_1_kwargs)
    sample_2_points = sample_2(**sample_2_kwargs)

    return np.hstack((sample_1_points, sample_2_points))


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
    random_state = check_random_state(random_state)

    vec = random_state.randn(n, d)
    vec *= radius / np.linalg.norm(vec, axis=1)[:, None]

    radial = random_state.rand(n, 1)
    vec *= radial ** (1 / d)

    if center == []:
        center = np.array([0] * d)
    vec = center + radius * vec

    return vec


def hyperSphere(n, d, radius=1.0, center=[], random_state=None):
    """
    Generates a sample from a uniform distribution on the hypersphere

    Parameters
    ----------
    n: int
        Number of data points.
    d: int
        Dimension of the hypersphere
    radius: float
        Radius of the hypersphere
    center: list, tuple, np.array
        Center of the hypersphere
    random_state: int, np.random.RandomState instance
        Random number generator

    Returns
    -------
    data: np.array, (npoints x ndim)
        Generated data
    """
    random_state = check_random_state(random_state)
    if center == []:
        center = np.array([0] * (d + 1))
    vec = random_state.randn(n, d + 1)
    vec /= np.linalg.norm(vec, axis=1)[:, None]
    vec = center + radius * vec
    return vec


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
    _height = height * np.prod(np.sin(2 * np.pi * base_coord), axis=1, keepdims=True)
    return np.hstack((base_coord, _height))


def hyperTwinPeaksUniform(n, d=2, height=1.0, random_state=None):
    """
    Generates a uniform sample from a plane with protruding peaks.

    Translated from Kerstin Johnsson's R package intrinsicDimension, with amendments to ensure sample is uniform with respect to the Hausdorff measure.

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
    data = np.ndarray((0, d + 1))
    while data.shape[0] < n:
        num_to_sample = n - data.shape[0]
        base_coord = random_state.uniform(size=(num_to_sample, d))
        _height = height * np.prod(
            np.sin(2 * np.pi * base_coord), axis=1, keepdims=True
        )
        sampled_points = np.hstack((base_coord, _height))
        metric_matrix = np.ndarray(shape=(num_to_sample, d, d))
        for i in range(num_to_sample):
            for j in range(d):
                for k in range(d):
                    metric_matrix[i, j, k] = (
                        height
                        * np.prod(np.sin(2 * np.pi * np.delete(base_coord[i, :], [j])))
                        * np.prod(np.sin(2 * np.pi * np.delete(base_coord[i, :], [k])))
                        * np.cos(2 * np.pi * base_coord[i, j])
                        * np.cos(2 * np.pi * base_coord[i, k])
                        * 4
                        * np.power(np.pi, 2)
                        + np.eye(d)[j, k]
                    )
        volume_elements = np.zeros(num_to_sample)
        for _ in range(volume_elements.shape[0]):
            volume_elements[_] = np.linalg.det(metric_matrix[_])
        volume_elements = np.power(volume_elements, 0.5)
        test_numbers = (
            random_state.rand(num_to_sample) * height * 4 * np.power(np.pi, 2)
        ) + 1
        retained_points = sampled_points[(volume_elements > test_numbers), :]
        data = np.append(data, retained_points, axis=0)

    return data


def lineDiskBall(n, random_state=None):
    """
    Generates a sample from a uniform distribution on a line, an oblong disk and an oblong ball
    Translated from ldbl function in Hideitsu Hino's package

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
        (
            random_state.uniform(-1, 1, (13 * n, 2)),
            np.zeros(13 * n)[:, None],
        )
    )
    disc = disc[~(np.sqrt(np.sum(disc**2, axis=1)) > 1), :]
    disc = disc[:, [0, 2, 1]]
    disc[:, 2] = disc[:, 2] - min(disc[:, 2]) + max(line[:, 2])

    fb = random_state.uniform(-0.5, 0.5, size=(n * 100, 3))
    rmID = np.where(np.sqrt(np.sum(fb**2, axis=1)) > 0.5)[0]

    if len(rmID) > 0:
        fb = fb[~(np.sqrt(np.sum(fb**2, axis=1)) > 0.5), :]

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
            return np.sqrt(b**2 + omega**2 * (a + b * r) ** 2)

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
    if not r1 > r2:
        raise ValueError("The radii must satisfy r1 > r2.")
    phi_sample = hyperBall(n=n, d=1, radius=2 * np.pi, random_state=random_state).reshape([-1,1])
    u = np.hstack((r1 * np.cos(phi_sample), r1 * np.sin(phi_sample), np.zeros([n, 1])))
    v = r2 * (
        np.cos(phi_sample * n_twists) * u
        + np.sin(phi_sample * (n_twists))
        * np.hstack((np.zeros([n, 2]), np.ones([n, 1])))
    )
    return u + v


def torus(n, r1=1.0, r2=0.25, random_state=None):
    """Generates a uniform sample from the 2d torus.

    Parameters
    ----------
    n: int
        Number of data points to sample.
    r1: float, default=1.0
        Radius of central circle
    r2: float, default=0.25
        Radius of torus, must satisfy r2 < r1.

    Returns
    -------
    data: np.array, (npoints x ndim)
        Generated data
    """
    if not r1 > r2:
        raise ValueError("The radii must satisfy r1 > r2.")
    random_state = check_random_state(random_state)

    data = np.ndarray((0, 3))
    while data.shape[0] < n:
        num_to_sample = n - data.shape[0]

        phi_sample = random_state.rand([num_to_sample,1]) * 2 * np.pi
        theta_sample = random_state.rand([num_to_sample,1]) * 2 * np.pi

        u = np.hstack(
            (
                r1 * np.cos(phi_sample),
                r1 * np.sin(phi_sample),
                np.zeros([num_to_sample, 1]),
            )
        )
        v = np.hstack(
            (
                r2 * np.cos(phi_sample) * np.cos(theta_sample),
                r2 * np.sin(phi_sample) * np.cos(theta_sample),
                r2 * np.sin(theta_sample),
            )
        )
        sampled_points = u + v

        area_element = r1 + r2 * np.cos(theta_sample).reshape((num_to_sample,))
        test_numbers = random_state.rand(num_to_sample) * (r1 + r2)
        retained_points = sampled_points[(area_element > test_numbers), :]

        data = np.append(data, retained_points, axis=0)
    return data


def lorenz_attractor(n=10000, tmax=100.0):
    """Generates a sample from the Lorenz attractor.

    This method incorporates code by Christian Hill, January 2016 and January 2021
    https://github.com/scipython/scipython-maths/blob/master/lorenz/lorenz.py

    Parameters
    ----------
    n: int
        Number of data points to sample.
    tmax: float, default=100.0
        Time to run dynamical system

    Returns
    -------
    data: np.array, (npoints x ndim)
        Generated data
    """
    sigma, beta, rho = 10, 2.667, 28
    u0, v0, w0 = 0, 1, 1.05

    def lorenz(t, X, sigma, beta, rho):
        """The Lorenz equations."""
        u, v, w = X
        up = -sigma * (u - v)
        vp = rho * u - v - u * w
        wp = -beta * w + u * v
        return up, vp, wp

    t_eval = np.linspace(0, tmax, n)
    soln = solve_ivp(
        lorenz,
        (0, tmax),
        (u0, v0, w0),
        t_eval=t_eval,
        args=(sigma, beta, rho),
        dense_output=True,
    )
    return np.transpose(soln.y)


def dumbbell(n, connecting_radius=0.1, random_state=None):
    """Create a sample from a PL model of a 2-dimensional dumbell in R^3e.

    Parameters
    n   int
        number of points to sample
    connecting_radius    float
        radius of the connecting tube
    random_state: int, np.random.RandomState instance
        Random number generator
    """

    def profile_function(x):
        """Defines a dumbbell shape on [-1.0, 1.0] with radius 0.5 at the ends and connecting_radius at the connecting bar."""
        outer_x = 0.4
        inner_x = 0.4
        outer_slope = 0.5 / outer_x
        inner_slope = (0.5 - connecting_radius) / inner_x

        out_of_bounds_message = (
            f"The dumbbell is only defined between -1.0 and 1.0 but {x} was passed."
        )

        if x < -1.0:
            raise ValueError(out_of_bounds_message)
        elif x < -1.0 + outer_x:
            return outer_slope * (x + 1.0)
        elif x < -1.0 + outer_x + inner_x:
            return 0.5 - inner_slope * (x + 1.0 - outer_x)
        elif x < 1.0 - outer_x - inner_x:
            return connecting_radius
        elif x < 1.0 - outer_x:
            return connecting_radius + inner_slope * (x - 1.0 + outer_x + inner_x)
        elif x <= 1.0:
            return 0.5 - outer_slope * (x - 1.0 + outer_x)
        else:
            raise ValueError(out_of_bounds_message)

    dumbbell_surface = SurfaceOfRevolution(profile_function, random_state=random_state)
    return dumbbell_surface.sample_points(n=n)


class HilbertCurve:
    """Generates a dataset sampled from the Hilbert curve in a cube.

    Based on Kyle Finn's javascript code at https://stackoverflow.com/questions/14519267/

    Methods
    -------
    generate(n, depth=2)
        Generates n of data from a curve with recursion of the given depth.
    """

    def __init__(self, random_state=None):
        """
        Parameters
        ----------
        random_state: int
        """
        self.random_state = check_random_state(random_state)
        self._vertices = np.ndarray(shape=(0, 3))

    @staticmethod
    def _translate(s, point, d1, d2, d3):
        """
        Moves the point to the next sub-cube.
        """
        point = point - s * np.where(d1 < 0, d1, 0)
        point = point - s * np.where(d2 < 0, d2, 0)
        point = point - s * np.where(d3 < 0, d3, 0)
        return point

    def _make_curve(
        self,
        side,
        point=np.array([[0, 0, 0]]),
        d1=np.array([[1, 0, 0]]),
        d2=np.array([[0, 1, 0]]),
        d3=np.array([[0, 0, 1]]),
    ):
        """
        Recursively creates a list of vertices until side_length is 1.
        """
        if side == 1:
            self._vertices = np.append(self._vertices, point, axis=0)
        else:
            side /= 2
            point = self._translate(side, point, d1, d2, d3)
            self._make_curve(side, point, d2, d3, d1)
            self._make_curve(side, point + side * d1, d3, d1, d2)
            self._make_curve(side, point + side * (d1 + d2), d3, d1, d2)
            self._make_curve(side, point + side * d2, -d1, -d2, d3)
            self._make_curve(side, point + side * (d2 + d3), -d1, -d2, d3)
            self._make_curve(side, point + side * (d1 + d2 + d3), -d3, d1, -d2)
            self._make_curve(side, point + side * (d1 + d3), -d3, d1, -d2)
            self._make_curve(side, point + side * d3, d2, -d3, -d1)

    def _point_from_parameter(self, parameter):
        """Maps a parameter in the unit interval to a point on the curve"""
        parameter = parameter * (self._vertices.shape[0] - 1)
        fractional, floor = np.modf(parameter)
        start, end = self._vertices[int(floor), :], self._vertices[int(floor) + 1, :]
        point = start + (end - start) * (fractional)
        return point

    def generate(self, n, depth=2):
        """Generates points on a Hilbert curve.

        Parameters
        ----------
        n: int
            The number of points to generate.
        depth: int, default=2
            The number of recursive steps to take in approximating the curve.

        Returns
        -------
        data: np.array, (n x 3)
            Generated data.
        """
        if not isinstance(depth, int) or depth < 1:
            raise TypeError("The variable 'depth' must be a positive integer.")
        side_length = np.power(2, depth)
        self._make_curve(side_length)
        parameters = self.random_state.rand(n, 1)
        data = np.apply_along_axis(self._point_from_parameter, 1, parameters)
        return data

def mnist(imshape=False, digit_class=True):
    """Fetch MNIST dataset of 70000 28x28 greyscale handwritten digits from OpenML.
    Parameters
    ----------
        imshape: bool
            If true, return image data as 70000 28x28 numpy array. Defaults to flat 60000x748 array.
        digit_class: bool
            If true, return class of digits as well

    Returns
    -------
    X: np.array, (70000 x 748)
    y: np.array, (70000,)
    """
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    if imshape: X = X.reshape([-1, 28, 28])
    if digit_class:
        return X,y
    else:
        return X

class BenchmarkManifolds:
    """
    Generates a commonly used benchmark set of synthetic manifolds with known intrinsic dimension described by Hein et al. and Campadelli et al. [Campadelli2015]_

    Parameters
    ----------
    noise_type : str, 'uniform' or 'gaussian'
        Type of noise to generate
    """

    # class modified and adapted from https://github.com/stat-ml/GeoMLE
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
            [
                np.sin(t),
                p,
                np.sign(t) * (np.cos(t) - 1),
                np.zeros((dim - d - 1, n)),
            ]
        ).T
        assert data.shape == (n, dim)
        return data

    def _gen_sphere_data(self, n, dim, d):
        assert d < dim

        V = self.random_state.randn(n, d + 1)
        data = np.hstack(
            [
                V / np.sqrt((V**2).sum(axis=1))[:, None],
                np.zeros((n, dim - d - 1)),
            ]
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
                p1**2 * np.cos(2 * np.pi * p0),
                p2**2 * np.sin(2 * np.pi * p0),
                p1 + p2 + (p1 - p3) ** 2,
                p1 - 2 * p2 + (p0 - p3) ** 2,
                -p1 - 2 * p2 + (p2 - p3) ** 2,
                p0**2 - p1**2 + p2**2 - p3**2,
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
        X = np.hstack([X, (X**2).sum(axis=1)[:, np.newaxis]])
        data = np.hstack([X, np.sin(X), X**2])

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
    """Implementation of surfaces of revolution."""

    def __init__(self, func, how="x", xmin=-1.0, xmax=1.0, random_state=None):
        """
        Create a surface of revolution which is symmetric around the x-axix.

        :param func: a callable which is the real-valued function to be rotated, y=f(x)
        :param how: a string indicating which axis to rotate the graph around, default is "x"
        :param xmin: the left boundary of the surface
        :param xmax: the right boundary of the surface
        """
        if not xmin < xmax:
            raise ValueError("The variable xmin must be less than xmax")
        if not how in ["x", "y"]:
            raise ValueError("The variable how must be 'x' or 'y'.")
        self.func = func
        self.how = how
        self.xmin = xmin
        self.xmax = xmax
        self.random_state = check_random_state(random_state)
        (
            self.xvals,
            self.fvals,
            self.f_prime_vals,
            self.jacobian_vals,
        ) = self._fetch_function_values()

    def plot_curve(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if self.how == "x":
            ax.scatter(self.xvals, self.fvals, c="r", marker="o")
            ax.scatter(self.xvals, self.jacobian_vals, c="g", marker="o")
        else:
            ax.scatter(self.fvals, self.xvals, c="r", marker="o")
            ax.scatter(self.fvals, self.jacobian_vals, c="g", marker="o")
        plt.show()

    def _fetch_function_values(self):
        """
        Calculate the values of the function, the derivative and the Jacobian on a discrete grid.
        The results are stored in class variables.
        If how="y" then f_prime_vals = dx/dy and jacobian_vals are the derivatives of the inverse function
        :return xvals: the discrete grid of x-values
        :return fvals: the accompanying values of the function
        :return f_prime_vals: the values of the derivative dy/dx at each point, unless "how"="y", when it is dx/dy
        :return jacobian_vals: the scale of the volume element in terms of dx dtheta at x, or dy dtheta if "how"="y"
        """
        number_intervals = 20000
        xvals = np.linspace(self.xmin, self.xmax, number_intervals + 1)
        fvals = np.array([self.func(x) for x in xvals])
        if np.min(fvals) < 0:
            print(
                "Warning: the function describing the surface of revolution has negative values"
            )

        if self.how == "x":
            f_prime_vals = np.gradient(fvals, xvals)
            jacobian_vals = fvals * np.sqrt(1 + np.square(f_prime_vals))
        else:
            f_prime_vals = -1 / np.gradient(fvals, xvals)
            jacobian_vals = xvals * np.sqrt(1 + np.square(f_prime_vals))

        return xvals, fvals, f_prime_vals, jacobian_vals

    def sample_points(self, n=10000):
        """
        Sample n points from the surface of revolution.
        :param n: the number of points to sample
        :return: an array of shape (n, 3), the sampled points.
        """
        jac_max = np.max(self.jacobian_vals)
        jac_min = np.min(self.jacobian_vals)
        if jac_min < jac_max:
            accepted_x = []
            while len(accepted_x) < n:
                points_to_calculate = n - len(accepted_x)
                generated_x = self.random_state.uniform(
                    self.xmin, self.xmax, size=points_to_calculate
                )
                jacobian = np.interp(generated_x, self.xvals, self.jacobian_vals)
                rejection_parameter = np.random.uniform(0, jac_max, points_to_calculate)
                accepted_x = np.concatenate(
                    [accepted_x, generated_x[rejection_parameter < jacobian]]
                )
        else:
            accepted_x = self.random_state.uniform(self.xmin, self.xmax, size=n)
        theta = np.random.uniform(0, 2 * np.pi, n)
        f = np.interp(accepted_x, self.xvals, self.fvals)

        if self.how == "x":
            y = f * np.cos(theta)
            z = f * np.sin(theta)
            return np.stack((accepted_x, y, z), axis=1)
        else:
            x = accepted_x * np.cos(theta)
            z = accepted_x * np.sin(theta)
            return np.stack((x, f, z), axis=1)
