[![Tests](https://github.com/scikit-learn-contrib/scikit-dimension/actions/workflows/tests.yml/badge.svg)](https://github.com/scikit-learn-contrib/scikit-dimension/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/scikit-dimension/badge/?version=latest)](https://scikit-dimension.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/scikit-dimension)](https://pypi.org/project/scikit-dimension)
[![License](https://img.shields.io/github/license/scikit-learn-contrib/scikit-dimension)](https://github.com/scikit-learn-contrib/scikit-dimension/blob/master/LICENSE)
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
git clone https://github.com/scikit-learn-contrib/scikit-dimension
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

### Citing and credits

Please cite Bac et al. (2021), *Scikit-Dimension: A Python Package for Intrinsic Dimension Estimation*, Entropy 23(10):1368, [doi:10.3390/e23101368](https://doi.org/10.3390/e23101368) (see `CITATION.cff`), together with the paper of each method you use (listed in its docstring).
Contributors are listed in `AUTHORS.md`; code adapted from other projects and its licences are listed in `LICENSES_bundled.txt`.

Related packages: [DADApy](https://github.com/sissa-data-science/DADApy) (Glielmo et al., Patterns 2022), [intrinsicDimension](https://cran.r-project.org/package=intrinsicDimension), [Rdimtools](https://cran.r-project.org/package=Rdimtools), [PCAtools](https://bioconductor.org/packages/PCAtools).
