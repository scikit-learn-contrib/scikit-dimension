function [ betas ] = compute_betas( max_dim )
%COMPUTE_BETAS Computes betas between one and max_dim. This code is based on the paper 
%   
%   Díaz, M., Quiroz, A., & Velasco, M. (2018). 
%   Local angles and dimension estimation from data on manifolds.
% 
%   betas = compute_betas( max_dim ) Computes the variance of the angle between two Gaussian
%   vectors in RR^d, it does so for each d between 1 and max_dim.
betas = zeros(max_dim-2,1);
betas(1:2) = [pi^2/12; pi^2/4 - 2];
betas = betas(:);
if (max_dim >= 4) 
    for ii = 4:max_dim
        betas(ii-1) = betas(ii-3) - 2/(ii-2)^2;
    end
end
betas = [pi^2/4; betas];
end

