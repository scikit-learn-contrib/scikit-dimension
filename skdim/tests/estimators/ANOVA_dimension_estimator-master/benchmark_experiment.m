%% Benchmark experiment 
%   This code is based on the paper 
%   
%   Díaz, M., Quiroz, A., & Velasco, M. (2018). 
%   Local angles and dimension estimation from data on manifolds.
%
%   In order to run this experiment you need to download the code for DANCo
%   and the library manifolds that can be found in the following links:
%
%   https://www.mathworks.com/matlabcentral/fileexchange/40112-intrinsic-dimensionality-estimation-techniques?focused=3785375&tab=function 
%   http://www.ml.uni-saarland.de/code/IntDim/IntDim.htm
%
%   You will need to add compile the mex-file for generating the manifolds
%   and then add the folders with all the code aforementioned. 

clc; clear all; close all;
T = 50;
nb_manifolds = 13;
nb_estimators = 4;
MPE = zeros(nb_estimators, nb_manifolds);
MSE = zeros(nb_estimators, nb_manifolds);
estimated_dim = zeros(nb_estimators, nb_manifolds);

correct_dimensions = [9, 3, 4, 4, 2, 6, 2, 12, 20, 9, 2, 10, 1];
dimensions = [10, 5, 6, 8, 3, 36, 3, 72, 20, 10, 3, 10, 10];
for jj = 1:nb_manifolds
    fprintf('Testing manifold %d \n', jj); 
    d_correct = correct_dimensions(jj);
    for ii = 1:1
        data_ind = jj;
        X = GenerateManifoldData(data_ind,dimensions(jj),2500);
        [d_angular, d_angular_heuristic ] = ANOVA_global_estimator(X, 'maxdimension', 100);
        d_LB = 0;
        for kk = 10:20
            d_LB = d_LB + MLE(X, 'k', kk)/11;
        end
         [inds,dists] = KNN(X,12 ,true);
         [d_danco,~] = DANCoFit(X,'inds',inds,'dists',dists);
%          [d_danco,~] = DANCo(X,'inds',inds,'dists',dists);
        estimated_dim(:, jj) = estimated_dim(:, jj) + [d_angular; d_angular_heuristic; d_LB; d_danco];
        MPE(:, jj) = MPE(:, jj) + abs([d_angular; d_angular_heuristic; d_LB; d_danco] - d_correct)/d_correct;  
        MSE(:, jj) = MSE(:, jj) + abs([d_angular; d_angular_heuristic; d_LB; d_danco] - d_correct).^2;  

    end
    estimated_dim(:, jj) = estimated_dim(:, jj)/T;
    MPE(:, jj) = 100*MPE(:, jj)/T;
    MSE(:, jj) = MSE(:, jj)/T;
end
mean(MSE,2)
mean(MPE,2)
%% Saves results
save results/benchmark_experiment

