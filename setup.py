#! /usr/bin/env python
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
)
