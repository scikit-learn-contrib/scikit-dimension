|Travis|_ |Appveyor|_ |Codecov|_ |CircleCI|_ |Documentation|_ |LanguageGrade|_ |License|_

.. |Travis| image:: https://travis-ci.com/j-bac/scikit-dimension.svg?branch=master
.. _Travis: https://travis-ci.com/j-bac/scikit-dimension
.. |Appveyor| image:: https://ci.appveyor.com/api/projects/status/tvumlfad69g6ap3u/branch/master?svg=true
.. _Appveyor: https://ci.appveyor.com/project/j-bac/scikit-dimension/branch/master
.. |Codecov| image:: https://codecov.io/gh/j-bac/scikit-dimension/badge.svg?branch=master&service=github
.. _Codecov: https://codecov.io/gh/j-bac/scikit-dimension/
.. |CircleCI| image:: https://circleci.com/gh/j-bac/scikit-dimension/tree/master.svg?style=shield
.. _CircleCI: https://circleci.com/gh/j-bac/scikit-dimension/tree/master
.. |Documentation| image:: https://readthedocs.org/projects/scikit-dimension/badge/?version=latest
.. _Documentation: https://scikit-dimension.readthedocs.io
.. |LanguageGrade| image:: https://img.shields.io/lgtm/grade/python/g/j-bac/scikit-dimension.svg?logo=lgtm&logoWidth=18
.. _LanguageGrade: https://lgtm.com/projects/g/j-bac/scikit-dimension/context:python
.. |License| image:: https://img.shields.io/github/license/j-bac/scikit-dimension.svg
.. _License: https://github.com/j-bac/scikit-dimension/blob/master/LICENSE

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
- generate toy datasets and widely used synthetic manifolds to benchmark estimators 

Support
^^^^^^^
Feel free to report an `issue <https://github.com/j-bac/scikit-dimension/issues/new/choose>`_

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Main

   installation
   quick_start
   api
   release_notes
   references
   contributing

.. toctree::
   :caption: Tutorials
   :maxdepth: 1
   :hidden:

   basics