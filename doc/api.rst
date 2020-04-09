####################
scikit-dimension API
####################

The module gid provides global intrinsic dimension estimators (i.e., one value for the entire dataset) and the module lid contains local intrinsic dimension estimators (i.e., functions provide values that are estimated in neighborhoods of each point).

lid (local intrinsic dimension)
===============================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   skdim.lid.ESS


.. autosummary::
   :toctree: generated/
   :template: class.rst

   skdim.lid.FisherS

.. autosummary::
   :toctree: generated/
   :template: class.rst

   skdim.lid.lPCA

.. autosummary::
   :toctree: generated/
   :template: class.rst

   skdim.lid.MiND_ML

.. autosummary::
   :toctree: generated/
   :template: class.rst

   skdim.lid.MOM

.. autosummary::
   :toctree: generated/
   :template: class.rst

   skdim.lid.TLE

gid (global intrinsic dimension)
================================

.. autosummary::
   :toctree: generated/
   :template: class.rst

    skdim.gid.CorrInt

.. autosummary::
   :toctree: generated/
   :template: class.rst

    skdim.gid.DANCo

.. autosummary::
   :toctree: generated/
   :template: class.rst

    skdim.gid.KNN

.. autosummary::
   :toctree: generated/
   :template: class.rst

    skdim.gid.Mada

.. autosummary::
   :toctree: generated/
   :template: class.rst

    skdim.gid.MLE

.. autosummary::
   :toctree: generated/
   :template: class.rst

    skdim.gid.TwoNN
