function id = ided(dists)
% dists - nearest-neighbor distances (k x 1, or k x n), sorted
    k = size(dists,1);
    n1 = floor(k/2);
    n2 = k;
    r1 = dists(n1,:);
    r2 = dists(n2,:);
    id = log(n1/n2) ./ log(r1./r2);
end
