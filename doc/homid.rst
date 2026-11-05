HomID datasets
==============

HomID provides five datasets for benchmarking intrinsic dimension estimators [Das2025]_. Each generator returns a dense ``float64`` array with one sample per row.

.. currentmodule:: skdim.datasets

.. autosummary::

   grassmannian_projector
   grassmannian_vector
   stiefel_matrix
   stiefel_vector
   flag_vector

Example
-------

.. code-block:: python

   from skdim import datasets

   data = datasets.grassmannian_projector(n=100, p=4, k=2, random_state=42)
   assert data.shape == (100, 16)

``n`` is the sample count, ``p`` is the underlying vector-space dimension,
and ``k`` is the subspace or frame rank. For flags, ``k1`` and ``k2`` are
nested prefix ranks. Matrix outputs are flattened in row-major order.
See the API pages above for valid parameters, output shapes and intrinsic
dimensions.

Pass an integer ``random_state`` to reproduce the original HomID samples
with matching parameters and NumPy, SciPy and BLAS/LAPACK versions. The
vector generators retain the original per-point seeds ``random_state + i``.
Seeds ``s`` and ``s + 1`` therefore share ``n - 1`` points; use seeds at least
``n`` apart for independent replicates.
For these generators, a ``RandomState`` instance or ``None`` selects one
base seed first. The matrix generators consume the supplied random stream
directly. See the function notes for details.

Vector feature counts grow combinatorially. Large requests may be slow and
require substantial memory.

Local caching
-------------

Use ``joblib.Memory`` to reuse generated data:

.. code-block:: python

   from joblib import Memory
   import skdim
   from skdim import datasets

   memory = Memory(f".cache/homid/{skdim.__version__}", verbose=0)
   generate = memory.cache(datasets.grassmannian_vector)
   data = generate(n=100, p=4, k=2, random_state=42)

Clear the cache after changing the generator source during development.
