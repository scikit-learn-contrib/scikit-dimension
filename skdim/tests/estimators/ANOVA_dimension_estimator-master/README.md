# ANOVA dimension estimator
This repository is based and supplements the paper 

>	   Díaz, M., Quiroz, A., & Velasco, M. (2018). Local angles and dimension estimation from data on manifolds.
    
## Estimators 
We implemented functions for both our local and global estimators and can be found in '*ANOVA_local_estimator.m*' and '*ANOVA_global_estimator.m*', respectively. Each one of these files contains instructions on how to use them. The user can set parameters such as the number of neighbors, number of centers, maximum possible dimension and whether to use the naive or the kernel-based estimator. 

## Experiments 
Additionally, we include code for the experiments described in the paper. That is, we have two scripts named '*benchmark_experiment.m*' and '*normal_conjecture_experiment.m*'. 

## Structure 
**/utils** This folder contains all the necessary functions to run the estimators. 

**/data** This folder includes a .mat file with the data to run the kernel-based estimator (The code will create it again if it doesn't exist).

**/results** The experiment scripts paste all their results here (both figures and .mat files).

## Dependencies

In order to run the **experiments** one need to add to path some dependencies.

1. To run the benchmark experiment one needs to add to path code for [DANCo](https://www.mathworks.com/matlabcentral/fileexchange/40112-intrinsic-dimensionality-estimation-techniques?focused=3785375&tab=function) and the [data generator](http://www.ml.uni-saarland.de/code/IntDim/IntDim.htm). 

1. One can run the last two experiments without any extra code, nonetheless we added sections that make better-than-standard plots and save them. To run those one needs to add to path code for [linspecer](https://github.com/davidkun/linspecer) and [export_fig](https://github.com/altmany/export_fig).
