function [ dimensions, first_moments,indicesknn, Us] = ANOVA_local_estimator( dataset, centers, varargin  )
% ANOVA_LOCAL_ESTIMATOR Estimates the dimension of the manifold in which
%   the dataset lies at the center points. This code is based on the paper 
%   
%   Díaz, M., Quiroz, A., & Velasco, M. (2018). 
%   Local angles and dimension estimation from data on manifolds.
%   
%   [d, first_moments ] = ANOVA_local_estimator( dataset, centers ) Estimate the dimension with
%   for each one of the center points (where the centers are taken as columns of the matrix centers ). 
%   Addtionally it gets an empirical expected value ('first_moments') of the angles between the neighbors of each center.  
%
%   [d, dh] = ANOVA_global_estimator( __ , Name, Value ) Estimate the dimension with
%   the user-specified parameters. One can vary the numerical parameters number of
%   neighbors 'k' and maximum possible dimension 'maxdimension'. The boolean parameter 'basic' determines what estimator to use between the basic
%   estimator or the kernel-based one. The boolean parameter 'verbose'
%   determines if the script prints a summary of the results or not.

% Adds all the necessary files to the path
folder = fileparts(which(mfilename)); 
addpath(genpath(folder));


if nargin < 2
    error('Please specify the data set and the centers');
end
[D, n] = size(dataset);
%% --------------------------------------
% Sets parameters to default
%----------------------------------------
params.D_max = D;
params.k = min(500,max(round(10*log10(n)), 10));
params.verbose = 0;
params.basic = 1;

%% --------------------------------------
% Retrieves user-specified parameters
%----------------------------------------
if (rem(length(varargin),2)==1)
    error('Options should be given in pairs');
else
    for itr=1:2:(length(varargin)-1)
        switch lower(varargin{itr})
            case 'maxdimension'
                params.D_max = varargin{itr+1};
            case 'k'
                params.k = varargin{itr+1};
            case 'verbose'
                params.verbose = varargin{itr+1};
            case 'basic'
                params.basic = varargin{itr+1};
            otherwise
                error(['Unrecognized option: ''' varargin{itr} '''']);
        end
    end
end

%% --------------------------------------
% Estimates the dimension
%----------------------------------------

% Allocations
dimensions = zeros(size(centers,2),1);
first_moments = zeros(size(centers,2),1);
Us = zeros(size(centers,2),1);
betas = compute_betas(params.D_max);


% Loads necessary data for the estimators
if ~params.basic
    try
        load(['data/angular_estimator_fit_' num2str(params.D_max) '_' num2str(params.k) '.mat']);
    catch
        M = 10000;
        data = zeros(M,params.D_max);
        for ii = 1:params.D_max
            data(:,ii) = simulate_statistic(ii,params.k,betas(ii));
        end
        file_name =['data/angular_estimator_fit_' num2str(params.D_max) '_' num2str(params.k) '.mat'];
        save(file_name,'data');
    end
else
    etas = compute_cuts(params.D_max);
end


% Estimates the dimension for each center
indicesknn = knnsearch(dataset',centers','K',params.k);
for ii = 1:length(dimensions)
    [U, first_moments(ii)] = compute_statistic(dataset(:,indicesknn(ii,:)), centers(:,ii));
    Us(ii) = U;
    if params.basic      
        dd = 1;
        while( dd <= length(etas) && U < etas(dd))
            dd=dd+1;
        end
        dimensions(ii) = dd;
    else
        max_density = 0;
        for jj =1:D
            dens = evaluate_approx_density(params.k*(U-betas(jj)),data(:,jj));
            if dens > max_density
                max_density = dens;
                dimensions(ii) = jj;
            end
        end
    end
    if params.verbose == 1
        fprintf('\t Center %d, dimension estimate %d \n', ii, dimensions(ii));
    end
end

end

