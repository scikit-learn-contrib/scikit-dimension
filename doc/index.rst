.. image:: https://github.com/scikit-learn-contrib/scikit-dimension/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/scikit-learn-contrib/scikit-dimension/actions/workflows/tests.yml
.. image:: https://readthedocs.org/projects/scikit-dimension/badge/?version=latest
   :target: https://scikit-dimension.readthedocs.io
.. image:: https://img.shields.io/pypi/v/scikit-dimension
   :target: https://pypi.org/project/scikit-dimension
.. image:: https://img.shields.io/github/license/scikit-learn-contrib/scikit-dimension
   :target: https://github.com/scikit-learn-contrib/scikit-dimension/blob/master/LICENSE

scikit-dimension - Intrinsic dimension estimation in Python
===========================================================

**scikit-dimension** is a Python module for intrinsic dimension estimation.

What is intrinsic dimension ?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Datasets available today frequently have a large number of features and are thus said to be high-dimensional. However many of these features are usually redundant, for example because of linear or non-linear correlations. In such a case it is possible to describe a dataset using a lower number of features. The notion of intrinsic dimension (ID) intuitively refers to the minimal number of features needed to represent a dataset with little information loss. 

In the context of the manifold hypothesis, i.e., the hypothesis that data are sampled from an underlying :math:`n`-dimensional manifold, the goal of ID estimation is to recover :math:`n`. Global estimators of ID are based on this hypothesis and return a single estimated dimension for the whole dataset.

When a dataset contains multiple manifolds or regions with different dimensionality (e.g., imagine sampling a line and a sphere and joining them as one dataset), they should be explored using local estimators. The idea behind local ID estimation is to operate in local neighborhoods of each point, where a manifold can be well approximated by its flat tangent space. These local pieces of data are assumed to be close to a uniformly distributed :math:`n`-dimensional ball. In this package, local datasets are simply defined using the :math:`k`-nearest neighbours of each point, and an ID value is thus returned for each datapoint. Such an approach also allows one to repurpose global estimators as local estimators by applying them in each local neighborhood. 

Organization of `scikit-dimension`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`scikit-dimension` provides a submodule `id` for intrinsic dimension estimators and `datasets` to generate benchmark datasets 

Functionalities
^^^^^^^^^^^^^^^
- estimate global intrinsic dimension
- estimate local intrinsic dimension
- choose the number of principal components (signal rank)
- generate toy datasets and widely used synthetic manifolds to benchmark estimators

Estimators
^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Family
     - Estimators
   * - Nearest-neighbour distances
     - ``TwoNN``, ``GRIDE``, ``MLE``, ``MOM``, ``TLE``, ``MADA``, ``KNN``
   * - Angles and norms
     - ``DANCo``, ``MiND_ML``, ``ESS``
   * - Correlation and topology
     - ``CorrInt``, ``PH``
   * - Separability
     - ``FisherS``
   * - PCA and signal rank
     - ``lPCA``: explained-variance rules (FO, Fan, maxgap, ratio, participation ratio, Kaiser, broken stick),
       random matrix theory (Marchenko-Pastur, Gavish-Donoho, Kritchman-Nadler), parallel analysis,
       Minka's PPCA evidence and scran's denoised PCA rule

Each estimator's docstring cites the method paper and, where one exists, the reference
implementation it is tested against (see :doc:`changelog`).

Citing
^^^^^^
If you use scikit-dimension, please cite Bac et al. (2021), *Scikit-Dimension: A Python Package for
Intrinsic Dimension Estimation*, Entropy 23(10):1368, `doi:10.3390/e23101368 <https://doi.org/10.3390/e23101368>`_,
and the paper of each method you use.

Support
^^^^^^^
Feel free to report an `issue <https://github.com/scikit-learn-contrib/scikit-dimension/issues/new/choose>`_

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Main

   installation
   quick_start
   basics
   api
   changelog
   references
   contributing

