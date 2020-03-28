Tight Local ID Estimator (TLE)
==============================

Welcome to the collection of Matlab scripts implementing intrinsic dimensionality (ID) estimators and experiments described in the paper:

Laurent Amsaleg, Oussama Chelly, Michael E. Houle, Ken-ichi Kawarabayashi, Miloš Radovanović and Weeris Treeratanajaru. Intrinsic dimensionality estimation within tight localities. In Proceedings of the SIAM International Conference on Data Mining (SDM), 2019.
http://perun.pmf.uns.ac.rs/radovanovic/tle/sdm19tle.pdf

and the accompanying supplement: http://perun.pmf.uns.ac.rs/radovanovic/tle/sdm19tle-supplement.pdf


Version history:
----------------

2019-05-02: Initial release.


Quick description of the scripts:
---------------------------------

id*.m:
  Implementations of various ID estimators.

genfig_meanvarbyd_outlierness_all.m:
  Generate synthetic-data plots from Figure 2 in the paper, and Figure 1 in the supplement.

genfig_meanvarbyk.m:
  Generate synthetic-data plots from Figure 3 in the paper.

genfig_boxplot_all2multik.m, genfig_boxplot_fun2multik.m:
  Generate real-data box plots from Figure 4 in the paper, and Figure 2 in the supplement. (Data sets available separately.)

genfig_mdata.m:
  Generate m-family synthetic-data plots from Figure 3 in the supplement. (Data sets available separately.)

compute_ids_dir_all.m, compute_ids_dir.m:
  Compute ID estimates for all data sets in given directories.

compute_knn_dir_all.m, compute_knn_dir.m:
  Compute k-NN distances and indexes for all data sets in given directories.

torusL2DistForKNNSearch.m:
  Modified Euclidean distance to enable k-NN search in the multidimensional torus setting (see supplement).

GetDim.mexw64:
  Matlab executable compiled for 64-bit Windows from matlab\GetDim.cpp by Matthias Hein and Jean-Yves Audibert. We use it to compute correlation dimension. See https://www.ml.uni-saarland.de/code/IntDim/IntDim.htm

All Matlab code written by Miloš Radovanović, with the help of (pseudo) code by Oussama Chelly and Michael E. Houle.


Contact:
--------

Questions? Comments? Please write to Miloš Radovanović <radacha@dmi.uns.ac.rs>


Licensing information:
----------------------

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License: http://creativecommons.org/licenses/by-sa/4.0/

For attribution, please cite the above paper.
