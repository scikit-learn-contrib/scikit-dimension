Quick Start
===========

Local and global estimators can be used in this way:

.. code-block:: python

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
