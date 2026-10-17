#! /usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""A package implementing algorithms to estimate local and global intrinsic dimension."""

import codecs
import os

from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join("skdim", "_version.py")
with open(ver_file) as f:
    exec(f.read())

DISTNAME = "scikit-dimension"
DESCRIPTION = "scikit-dimension is a Python module for intrinsic dimension estimation built according to the scikit-learn API and distributed under the 3-Clause BSD license.."
MAINTAINER = "Jonathan Bac"
URL = "https://github.com/j-bac/scikit-dimension"
LICENSE = "BSD-3-clause"
DOWNLOAD_URL = "https://github.com/j-bac/scikit-dimension"
VERSION = __version__
INSTALL_REQUIRES = ["numpy", "numba", "scipy", "scikit-learn", "matplotlib"]
CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "License :: OSI Approved",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
]
EXTRAS_REQUIRE = {
    "tests": ["pytest", "pytest-cov"],
    "docs": ["sphinx", "sphinx-gallery", "sphinx_rtd_theme", "numpydoc", "matplotlib",],
}

setup(
    name=DISTNAME,
    maintainer=MAINTAINER,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    zip_safe=False,  # the package can run out of an .egg file
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    package_data={"skdim.id": ["DANCoFit/*.npz"]},
)
