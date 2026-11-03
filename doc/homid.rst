HomID datasets
==============

HomID uses homogeneous spaces to test intrinsic dimension estimation under
anisotropic embeddings [Das2025]_. These generators sample the five HomID
constructions on demand. They return dense ``float64`` arrays with one sample
per row. They perform no file I/O and do not run an ID estimator.

Generate a small dataset
------------------------

Here, ``n`` is the sample count. ``p`` is the order of the ambient orthogonal
group. ``k`` is the frame width or subspace dimension. It is not a neighborhood
size. ``p`` is also not the flattened feature count.

.. code-block:: python

   import numpy as np
   from skdim import datasets

   data = datasets.grassmannian_projector(n=100, p=4, k=2, random_state=0)
   assert data.shape == (100, 16)
   assert data.dtype == np.float64

   # An integer seed reproduces the same samples.
   repeated = datasets.grassmannian_projector(n=100, p=4, k=2, random_state=0)
   np.testing.assert_array_equal(data, repeated)

The corresponding vector embedding has the same local dimension and a
different ambient representation:

.. code-block:: python

   vectors = datasets.grassmannian_vector(n=100, p=4, k=2, random_state=0)
   assert vectors.shape == (100, 6)

Pass an integer seed for repeatable calls. A ``numpy.random.RandomState``
instance is consumed as a random stream. With ``random_state=None``, the
functions use the NumPy random stream selected by scikit-learn's
``check_random_state`` convention. They do not reset the global seed.

Constructions and dimensions
----------------------------

Let :math:`C(p,k)=\binom{p}{k}`. The dimensions below describe the support
of the noise-free samples. Each output shape includes the sample count ``n``.

.. list-table::
   :header-rows: 1
   :widths: 28 24 23 25

   * - Generator
     - Construction
     - Output shape
     - Local intrinsic dimension
   * - :func:`~skdim.datasets.grassmannian_projector`
     - Orthogonal rank-:math:`k` projector
     - :math:`(n,p^2)`
     - :math:`k(p-k)`
   * - :func:`~skdim.datasets.grassmannian_vector`
     - Unit Plücker vector
     - :math:`(n,C(p,k))`
     - :math:`k(p-k)`
   * - :func:`~skdim.datasets.stiefel_matrix`
     - Orthonormal :math:`p\times k` frame
     - :math:`(n,pk)`
     - :math:`pk-k(k+1)/2`
   * - :func:`~skdim.datasets.stiefel_vector`
     - Plücker vector concatenated with an :math:`SO(k)` matrix
     - :math:`(n,C(p,k)+k^2)`
     - :math:`pk-k(k+1)/2`
   * - :func:`~skdim.datasets.flag_vector`
     - Tensor product of Plücker vectors from disjoint blocks
     - :math:`(n,C(p,k_1)C(p,k_2))`
     - :math:`(k_1+k_2)p-k_1^2-k_2^2-k_1k_2`

All generators accept integer ``n >= 0`` and ``p >= 2``. The first four require
``1 <= k < p``. The flag generator requires positive integer block sizes
``k1`` and ``k2`` with ``k1 + k2 <= p``. Equal block sizes are allowed. The
remaining block may have size zero. With ``n=0``, the output has zero rows and
the same feature count.

Matrix-valued constructions are flattened in row-major order. For example,
``stiefel_matrix(n, p, k)`` can be reshaped to ``(n, p, k)``. For the flag
construction, the block ranks are disjoint sizes, not nested subspace ranks.

.. code-block:: python

   frames = datasets.stiefel_matrix(n=100, p=4, k=2, random_state=0)
   frame_vectors = datasets.stiefel_vector(n=100, p=4, k=2, random_state=0)
   flags = datasets.flag_vector(n=100, p=5, k1=2, k2=2, random_state=0)
   assert frames.shape == (100, 8)
   assert frame_vectors.shape == (100, 10)
   assert flags.shape == (100, 100)

Support and topology
--------------------

The vector Grassmannian construction retains the sign of the unit Plücker
vector. It represents an oriented Grassmannian. The projector construction
forgets this orientation. Their local intrinsic dimensions agree.

``stiefel_vector`` follows the paper's product construction. Its support is an
oriented Grassmannian times :math:`SO(k)`. It has the Stiefel local dimension,
but it is not generally globally the Stiefel manifold. For example, at
``p=3, k=2`` its support is :math:`S^2\times S^1`, whereas the Stiefel frame
space is :math:`SO(3)`. At ``k=1``, :math:`SO(1)=\{1\}` is the constant factor.

``flag_vector`` uses the same orthogonal frame for its two disjoint blocks.
The outer product identifies :math:`(a,b)` and :math:`(-a,-b)`, where ``a``
and ``b`` are the block Plücker vectors. Thus it represents a finite quotient
of the separately oriented block configurations. This identification does
not change the local intrinsic dimension. The map should not be interpreted
as an injective embedding of the full separately oriented flag space.

Differences from the archived HomID code
----------------------------------------

These functions implement the paper definitions. They are not a promise of
sample-by-sample agreement with the archived research scripts.

* Gaussian QR factors receive the diagonal sign correction needed for Haar
  sampling. Projectors are unchanged by column-sign choices. Frame-valued
  samples are sensitive to those choices.
* The additional rotation in ``stiefel_vector`` is drawn from :math:`SO(k)`.
  The archived implementation used :math:`O(k)`.
* ``flag_vector`` uses disjoint frame blocks. The archived implementation used
  overlapping prefixes. These are different constructions in general.
* The functions use one ``random_state`` stream per call. They do not use the
  archived per-sample global reseeding scheme.

The original HomID repository and its archived results are unchanged.

Cost and optional local caching
-------------------------------

The vector feature counts grow with binomial coefficients. The flag feature
count is their product. Check the output shape before choosing parameters.
A dense ``float64`` result alone requires ``8 * n * n_features`` bytes.
QR factorizations and determinant calculations require additional work and
temporary memory. Caching does not reduce the memory needed to load a result.

For ``n > 0``, generation emits a ``UserWarning`` if the output has at least
10,000 features or requires at least 256 MiB of dense storage. A size guard
rejects requests beyond NumPy's index capacity with ``ValueError``. Smaller
requests can still exceed the available memory.

Generation is local and on demand. There is no automatic cache or external
dataset download. To reuse an expensive result, a caller can use
``joblib.Memory`` with an explicit integer seed:

.. code-block:: python

   from joblib import Memory
   import skdim
   from skdim import datasets

   memory = Memory(location=f".cache/homid/{skdim.__version__}", verbose=0)
   cached_generator = memory.cache(datasets.grassmannian_vector)
   data = cached_generator(n=100, p=4, k=2, random_state=0)

This optional example writes only to the caller-selected cache directory.
The versioned path separates package versions. Clear the cache after changing
source code during development. Reusing a cache is a local workflow choice;
it is not part of the generator API.
