[![Build Status](https://travis-ci.com/j-bac/skcontrib-id-estimators.svg?token=St4VFQxgQ93UPzzi8TFA&branch=master)](https://travis-ci.com/j-bac/skcontrib-id-estimators)
[![Build status](https://ci.appveyor.com/api/projects/status/tvumlfad69g6ap3u/branch/master?svg=true)](https://ci.appveyor.com/project/j-bac/scikit-dimension/branch/master)
[![codecov](https://codecov.io/gh/j-bac/skcontrib-id-estimators/branch/master/graph/badge.svg)](https://codecov.io/gh/j-bac/skcontrib-id-estimators)
[![CircleCI](https://circleci.com/gh/j-bac/scikit-dimension/tree/master.svg?style=shield)](https://circleci.com/gh/j-bac/scikit-dimension/tree/master)
[![Documentation Status](https://readthedocs.org/projects/scikit-dimension/badge/?version=latest)](https://scikit-dimension.readthedocs.io/en/latest/?badge=latest)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/j-bac/skcontrib-id-estimators.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/j-bac/skcontrib-id-estimators/context:python)
[![GitHub license](https://img.shields.io/github/license/j-bac/scikit-dimension)](https://github.com/j-bac/scikit-dimension/blob/master/LICENSE)


# scikit-dimension

scikit-learn is a Python module for intrinsic dimension estimation built according to the [sscikit-learn](https://github.com/scikit-learn/scikit-learn) API and distributed under the 3-Clause BSD license.

### Installation

Using pip:
```bash
pip install git+https://github.com/j-bac/scikit-dimension.git
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

data = np.random.random((100,10))
ess = skdim.local_id.ESS().fit(data)
danco = skdim.global_id.DANCo().fit(data)
```

Please refer to the [documentation](https://scikit-dimension.readthedocs.io) for detailed API and examples.

### Credits and links to original implementations:

#### R
- Kerstin Johnsson
https://cran.r-project.org/web/packages/intrinsicDimension/index.html

- Hideitsu Hino
https://cran.r-project.org/web/packages/ider/index.html

#### MATLAB
- Gabriele Lombardi https://fr.mathworks.com/matlabcentral/fileexchange/40112-intrinsic-dimensionality-estimation-techniques
- Miloš Radovanović https://perun.pmf.uns.ac.rs/radovanovic/tle/

#### C++ TwoNN
- Elena Facco https://github.com/efacco/TWO-NN

#### Python TwoNN 
- Francesco Mottes https://github.com/fmottes/TWO-NN 
and my modified fork https://github.com/j-bac/TWO-NN
