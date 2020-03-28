.. project-template documentation master file, created by
   sphinx-quickstart on Mon Jan 18 14:44:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

##########
User guide
##########

What is intrinsic dimension ?
=============================

Datasets available today frequently have a large number of features and are thus said to be high-dimensional. However many of these features are usually redundant; for example they are often correlated in linear or non-linear fashion. In such a case it is possible to describe a dataset using a lower number of features. The notion of intrinsic dimension (ID) intuitively refers to the minimal number of features needed to represent a dataset with little information loss. 

In the context of the manifold hypothesis, i.e., the hypothesis that data are sampled from an underlying :math:`n`-dimensional manifold, the goal of ID estimation is to recover :math:`n`. Global estimators of ID are based on this hypothesis and return a single estimated dimension for the whole dataset.

When a dataset contains multiple manifolds or regions with different dimensionality (e.g., imagine sampling a line and a sphere and joining them as one dataset), they should be explored using local estimators. The idea behind local ID estimation is to operate in local neighborhoods of each point, where a manifold can be well approximated by its flat tangent space. These local pieces of data are assumed to be close to a uniformly distributed :math:`n`-dimensional ball. In this package, local datasets are simply defined using the :math:`k`-nearest neighbours of each point, and an ID value is thus returned for each datapoint. Such an approach also allows one to repurpose global estimators as local estimators by applying them in each local neighborhood. 

Organization of `scikit-dimension`
==================================

`scikit-dimension` provides a submodule `global_id` for global estimators and `local_id` for local estimators of ID. Both local and global estimators return a single value of ID given the input data, but local estimators assume the input data is a local piece of the whole dataset (i.e., the input data point cloud is close to a uniformly distributed :math:`n`-dimensional ball). The helper function skdim.asPointwise allows one to apply a local or global estimator to local datasets formed by the :math:`k`-nearest neighbours of each datapoint. Some global estimators, like maximum likelihood estimation (MLE), obtain a final global estimate by aggregating local estimates obtained in each point and thus also provide the option to return the local values of ID that they computed. 