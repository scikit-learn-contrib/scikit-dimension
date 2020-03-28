function [ V, E ] = compute_statistic( dataset, p )
% COMPUTE_STATISTIC Computes the estimate of the variance of the angles
%   between points of the form X-p where X belongs to the data set. This code is based on the paper 
%   
%   Díaz, M., Quiroz, A., & Velasco, M. (2018). 
%   Local angles and dimension estimation from data on manifolds.
%
%   [V, E] = compute_statistic( dataset, p ) Computes the empirical
%   variance V and expected value E of the angle between points in the
%   dataset centered at p.
%

k = size(dataset,2); 

dataset = dataset - repmat(p,1,k); %centers the points
if size(dataset,1) > 1
    dataset=dataset(:,any(dataset)); %deletes zero columns
end
norms = sqrt(sum(dataset.^2,1));
dot_prods = ((dataset'*dataset)./(norms'*norms)); %computes the normalized the cosines of the angles for every pair
angles = acos(dot_prods(triu(true(size(dot_prods)),1))); %gets the upper triangular part of the matrix of cosines and computes the angle
V = sum((angles - pi/2).^2/length(angles)); %computes the statistics
E = mean(angles-pi/2);
end

