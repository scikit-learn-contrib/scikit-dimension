.. automodule:: skdim

API
===

Import skdim as::

   import skdim

ID estimators
-------------

.. autosummary::
   :toctree: .

   id.CorrInt
   id.DANCo
   id.ESS
   id.FisherS
   id.GRIDE
   id.KNN
   id.lPCA
   id.MADA
   id.MiND_ML
   id.MLE
   id.MOM
   id.PH
   id.TLE
   id.TwoNN


Datasets
--------

.. autosummary::
   :toctree: .

   datasets.hyperSphere
   datasets.hyperBall
   datasets.hyperTwinPeaks
   datasets.lineDiskBall
   datasets.swissRoll3Sph
   datasets.BenchmarkManifolds
   datasets.product
   datasets.hyperTwinPeaksUniform
   datasets.toroidal_spiral
   datasets.torus
   datasets.lorenz_attractor
   datasets.dumbbell
   datasets.SurfaceOfRevolution


Embeddings
----------

.. autosummary::
   :toctree: .

   embeddings.monomial_embedding
   embeddings.delay_embedding
