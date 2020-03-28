function [  density ] = evaluate_approx_density( x, data )
%EVALUATE_APPROX_DENSITY Evaluates normal kernel at each component of x.
%
%   densities = evaluate_approx_density( x, data ) Evaluates the normal 
%   distribution estimated with "data" at the different components of the
%   vector x. 

M = length(data);
h = (4/(3*M))^(1/5);
densities = normpdf(x,data,h);
density = mean(densities)/h;

end

