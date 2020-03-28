function D2 = torusL2DistForKNNSearch(ZI, ZJ)
% function D2 = torusL2DistForKNNSearch(ZI, ZJ),
%  
% taking as arguments a 1-by-N vector ZI containing a single row of X or Y,
% an M2-by-N matrix ZJ containing multiple rows of X or Y, and returning an
% M2-by-1 vector of distances D2, whose Jth element is the distance between
% the observations ZI and ZJ(J,:).
range = 1;
N = length(ZI);
[M2,N2] = size(ZJ);
if N ~= N2, error('Argument dimension mismatch'); end
D2 = zeros(M2,1);
for j = 1:M2
    diff = abs(ZI-ZJ(j,:));
    L = min(diff,range-diff);
    L = L.*L;
    D2(j) = sqrt(sum(L));
end