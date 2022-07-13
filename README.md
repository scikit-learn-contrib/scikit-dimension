[![Build status](https://ci.appveyor.com/api/projects/status/tvumlfad69g6ap3u/branch/master?svg=true)](https://ci.appveyor.com/project/j-bac/scikit-dimension/branch/master)
[![CircleCI](https://circleci.com/gh/scikit-learn-contrib/scikit-dimension/tree/master.svg?style=shield)](https://app.circleci.com/pipelines/github/scikit-learn-contrib/scikit-dimension)
[![Documentation Status](https://readthedocs.org/projects/scikit-dimension/badge/?version=latest)](https://scikit-dimension.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/j-bac/scikit-dimension/branch/master/graph/badge.svg)](https://codecov.io/gh/j-bac/scikit-dimension)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/j-bac/scikit-dimension.svg?logo=lgtm&logoWidth=18&label=%20code%20quality)](https://lgtm.com/projects/g/j-bac/scikit-dimension/context:python)
[![GitHub license](https://img.shields.io/github/license/j-bac/scikit-dimension)](https://github.com/j-bac/scikit-dimension/blob/master/LICENSE)
[![Downloads](https://pepy.tech/badge/scikit-dimension)](https://pepy.tech/project/scikit-dimension)

# scikit-dimension

scikit-dimension is a Python module for intrinsic dimension estimation built according to the [scikit-learn](https://github.com/scikit-learn/scikit-learn) API and distributed under the 3-Clause BSD license.

Please refer to the [documentation](https://scikit-dimension.readthedocs.io) and the [paper](https://www.mdpi.com/1099-4300/23/10/1368) for detailed API, examples and references

### Installation

Using pip:
```bash
pip install scikit-dimension
```

From source:
```bash
git clone https://github.com/j-bac/scikit-dimension
cd scikit-dimension
pip install .
```
 
### Quick start

Local and global estimators can be used in this way:

```python
import skdim
import numpy as np

#generate data : np.array (n_points x n_dim). Here a uniformly sampled 5-ball embedded in 10 dimensions
data = np.zeros((1000,10))
data[:,:5] = skdim.datasets.hyperBall(n = 1000, d = 5, radius = 1, random_state = 0)

#estimate global intrinsic dimension
danco = skdim.id.DANCo().fit(data)
#estimate local intrinsic dimension (dimension in k-nearest-neighborhoods around each point):
lpca = skdim.id.lPCA().fit_pw(data,
                              n_neighbors = 100,
                              n_jobs = 1)
                            
#get estimated intrinsic dimension
print(danco.dimension_, np.mean(lpca.dimension_pw_)) 
```
