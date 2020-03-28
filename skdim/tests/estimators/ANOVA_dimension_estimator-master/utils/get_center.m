function [ indx ] = get_center( dataset )
% GET_CENTER Finds the most central point in the dataset. This code is based on the paper 
%   
%   Díaz, M., Quiroz, A., & Velasco, M. (2018). 
%   Local angles and dimension estimation from data on manifolds.
%
%   indx = get_center( dataset ) Returns the index of the most central point with respect
%   to a centrality measure described in the paper. 
[d,n] = size(dataset);
centr = zeros(1,n);
vcentral = 0.5 - abs(0.5-((2*(1:n)-1)/(2*n)));
for ii = 1:d
    [~, icentral] = sort(dataset(ii,:));
    centr(icentral) = centr(icentral) + vcentral;
end
indx = find(max(centr)==centr); 
indx = indx(1);
end

