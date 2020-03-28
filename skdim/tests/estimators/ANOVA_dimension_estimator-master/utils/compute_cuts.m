function [ etas ] = compute_cuts( max_dim )
%COMPUTE_CUTS Computes the cuts that define the thresholds to decide between
%   dimensions. This code is based on the paper 
%   
%   Díaz, M., Quiroz, A., & Velasco, M. (2018). 
%   Local angles and dimension estimation from data on manifolds.
%
%   etas = compute_cuts(max_dim) Computes the cuts to decide between any
%   dimension between 1 and max_dim.
    betas = compute_betas( max_dim );
    etas = zeros(length(betas)-1,1);
    for ii = 1:length(etas)
        etas(ii) = (betas(ii) + betas(ii+1))/2;
    end
end

