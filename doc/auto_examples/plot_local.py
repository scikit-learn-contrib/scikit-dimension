#
# MIT License
#
# Copyright (c) 2020 Jonathan Bac
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
#
"""
===========================
Local ID example
===========================
"""
import skdim
from skdim import local_id
import numpy as np
from matplotlib import pyplot as plt

X = np.random.random((1000, 10))


# one neighborhood
res = local_id.ESS().fit(X)
res = local_id.FisherS().fit(X)
res = local_id.MOM().fit(X)
res = local_id.MiND_ML().fit(X)
res = local_id.TLE().fit(X)
res = local_id.lPCA().fit(X)


# all datapoint neighborhoods
pw_id = skdim.asPointwise(
    X, local_id.FisherS().fit, n_neighbors=100, n_jobs=1)
