function dataset = simulate_statistic( d, k,beta_d, M)
% SIMULATE_STATISTIC Generates empirical draws of the normalized angular-variance
%   statistic. This code is based on the paper 
%   
%   Díaz, M., Quiroz, A., & Velasco, M. (2018). 
%   Local angles and dimension estimation from data on manifolds.
%
%   dataset = simulate_statistic(d, k, beta_d) Generates 10000 instances of
%   k(U - beta_d) in dimension d, where U is the angular-variance statistic
%   with k neighbors. 
%
%   dataset = simulate_statistic(d, k, beta_d, M) Generates M instances of
%   k(U - beta_d) in dimension d, where U is the angular-variance statistic
%   with k neighbors.

if nargin < 4
    M =  10000; 
end
dataset = zeros(M,1);
for ii = 1:M
    X = randn(d,k);    
    dataset(ii) = k*(compute_statistic(X, zeros(d,1)) - beta_d);
end
end

