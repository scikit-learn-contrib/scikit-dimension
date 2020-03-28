function [ d, dh ] = ANOVA_global_estimator( dataset , varargin )
% ANOVA_GLOBAL_ESTIMATOR Estimates the dimension of the manifold in which
%   the dataset lies. This code is based on the paper 
%   
%   Díaz, M., Quiroz, A., & Velasco, M. (2018). 
%   Local angles and dimension estimation from data on manifolds.
%   
%   [d, dh] = ANOVA_global_estimator( dataset ) Estimate the dimension with
%   the default parameters. The estimate dh uses a heuritic to select
%   centers, the estimate d doesn't use this heuristic. 
%
%   [d, dh] = ANOVA_global_estimator( dataset , Name, Value ) Estimate the dimension with
%   the user-specified parameters. One can vary the numerial parameters number of
%   neighbors 'k', the number of centers 'numbercenters', and maximum dimension 'maxdimension'. The boolean
%   parameter 'basic' determines what estimator to use between the basic
%   estimator or the kernel-based one. The boolean parameter 'verbose'
%   determines if the script prints a summary of the results or not.

folder = fileparts(which(mfilename)); 
addpath(genpath(folder));


if nargin < 1; error('The data set is required');
end
[D, n] = size(dataset);

%% --------------------------------------
% Sets parameters to default
%----------------------------------------
params.basic = 1;
params.k = min(500,max(round(10*log10(n)), 10));
params.verbose = 0;
params.nbr_centers = round(2*(log(n)));
params.D_max = D;
%% --------------------------------------
% Retrieves user-specified parameters
%----------------------------------------
if (rem(length(varargin),2)==1)
    error('Options should be given in pairs');
else
    for itr=1:2:(length(varargin)-1)
        switch lower(varargin{itr})
            case 'basic'
                params.basic = varargin{itr+1};
            case 'k'
                params.k = varargin{itr+1};
            case 'numbercenters'
                params.nbr_centers = varargin{itr+1};
            case 'maxdimension'
                params.D_max = varargin{itr+1};
            case 'verbose'
                params.verbose = varargin{itr+1};
            otherwise
                error(['Unrecognized option: ''' varargin{itr} '''']);
        end
    end
end

%% --------------------------------------
% Estimates the dimension
%----------------------------------------

%Gets the centers
ind_cen = zeros(params.nbr_centers,1);
for ii = 1:params.nbr_centers
    r_sample = randsample(n,ceil(n/(params.nbr_centers)));
    indx = get_center(dataset(:,r_sample));
    ind_cen(ii) = r_sample(indx);
end
centers = dataset(:,ind_cen);

%Gets local dimension estimates for each center
[estimates, first_moments] = ANOVA_local_estimator(dataset,centers, 'k', params.k, 'basic', params.basic, 'maxdimension', params.D_max);
[~, ind_heru] = sort(abs(first_moments));

%Combines the estimates
d = median(estimates);
dh = median(estimates(ind_heru(1:ceil(params.nbr_centers/2))));
if params.verbose
    fprintf('Estimate %d, estimate+heuristic: %d , number centers: %d, number of neighbors: %d\n', d, dh, nbr_centers, k);
end



end

