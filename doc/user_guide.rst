.. project-template documentation master file, created by
   sphinx-quickstart on Mon Jan 18 14:44:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

##########
User guide
##########

What is intrinsic dimension ?
=============================
Datasets available today frequently have a large number of features and are thus said to be high-dimensional. However these features are usually redundant; for example they are often correlated in linear or non-linear fashion. In such a case it is possible to describe a dataset using a lower number of features.
The notion of intrinsic dimension (ID) intuitively refers to the minimal number of features needed to represent data with little information loss. This concept is largely used but doesn't have a consensus mathematical definition. In the context of the manifold hypothesis, i.e., the hypothesis that data are sampled from an underlying :math:`n`-dimensional manifold, the goal of ID estimation is to recover :math:`n`. Global estimators of ID are designed based on this hypothesis and return a single estimated dimension for the whole dataset.

Datasets can also be composed of regions with different dimensionality (e.g., imagine sampling a line and a sphere and joining them as one dataset). Such datasets should be explored using local estimators, which estimate ID for each point by looking at its neighbourhood. The idea behind local ID estimation is to operate in local neighborhoods where a manifold can be well approximated by its flat tangent space. These local pieces of data are then close to a uniformly distributed :math:`n`-dimensional ball. In this package, local datasets are simply defined as the :math:`k`-nearest neighbours of each point. Such an approach also allows one to use global estimators as local estimators by applying them in each local neighborhood.

Organization of `scikit-dimension`
==================================

`scikit-dimension` provides a submodule `global_id` for global estimators and `local_id` for local estimators of ID. Both local and global estimators will return a single value of ID for the input data, but local estimators assume the input data is a local piece of the whole dataset (i.e., the point cloud is close to a uniformly distributed :math:`n`-dimensional ball). The helper function skdim.asPointwise allows one to apply a local or global estimator to local datasets formed by the :math:`k`-nearest neighbours of each datapoint. Some global estimators, maximum likelihood estimation (MLE), obtain a final global estimate by aggregating local estimates. Thus they also provide the option to return these local values of ID. 

References
==========


