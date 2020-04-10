#####################################
Quick Start with scikit-dimension
#####################################

``scikit-dimension`` provides Python implementations of scikit-learn compatible estimators. 

Installation
===================================================

To install with pip run::

    $ pip install git+https://github.com/j-bac/scikit-dimension.git

To install from source run::

    $ git clone https://github.com/j-bac/scikit-dimension
    $ cd scikit-dimension
    $ pip install .

Basic usage
===================================================

Local and global estimators can be used in this way:

.. code-block:: python

    import skdim
    import numpy as np

    #generate data : np.array (n_points x n_dim). Here a uniformly sampled 5-ball embedded in 10 dimensions
    data = np.zeros((1000,10))
    data[:,:5] = skdim.gendata.hyperBall(n_points = 1000, n_dim = 5, radius = 1, random_state = 0)

    #fit an estimator of global intrinsic dimension (gid)
    danco = skdim.gid.DANCo().fit(data)
    #fit an estimator of local intrinsic dimension (lid)
    fishers = skdim.lid.FisherS().fit(data)
    #fit a global or local estimator in k-nearest-neighborhoods of each point:
    lpca_pw = skdim.asPointwise(data = data,
                                class_instance = skdim.lid.lPCA(),
                                n_neighbors = 100,
                                n_jobs = 1)
                                
    #get estimated intrinsic dimension
    print(danco.dimension_, fishers.dimension_, np.mean(lpca_pw))