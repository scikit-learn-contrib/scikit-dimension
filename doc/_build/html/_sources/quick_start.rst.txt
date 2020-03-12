#####################################
Quick Start with scikit-dimension
#####################################

``scikit-dimension`` provides Python implementations of scikit-learn compatible estimators. 

Installation
===================================================

To install with pip run::

    $ git clone https://github.com/scikit-learn-contrib/project-template.git

To install from source run::

    $ git clone https://github.com/scikit-learn-contrib/project-template.git

Basic usage
===================================================

Global intrinsic dimension estimators

.. code-block:: python

    from skdim import global_id
    import numpy as np
    from matplotlib import pyplot as plt

    X=np.random.random((1000,10))

    res = global_id.CorrInt().fit(X).dimension_
    res = global_id.DANCo().fit(X).dimension_
    res = global_id.KNN().fit(X).dimension_
    res = global_id.Mada().fit(X).dimension_
    res = global_id.MLE().fit(X).dimension_
    res = global_id.TwoNN().fit(X).dimension_

Local intrinsic dimension estimators

.. code-block:: python

    import skdim
    from skdim import local_id
    import numpy as np
    import matplotlib.pyplot as plt

    X = np.random.random((1000,10))

    #one neighborhood
    res = local_id.ESS().fit(X)
    res = local_id.FisherS().fit(X)
    res = local_id.MOM().fit(X)
    res = local_id.MiND_ML().fit(X)
    res = local_id.TLE().fit(X)
    res = local_id.lPCA().fit(X)


    #all datapoint neighborhoods
    pw_id = skdim.commonfuncs.asPointwise(X,local_id.FisherS().fit,n_neighbors=100,n_jobs=1)
