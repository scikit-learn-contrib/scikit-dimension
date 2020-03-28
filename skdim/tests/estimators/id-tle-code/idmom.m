function id = idmom(dists)
% dists - nearest-neighbor distances (k x 1, or k x n), sorted
    k = size(dists,1);
    w = dists(k,:);
    m1 = sum(dists) / k;
    id = -m1 ./ (m1-w);
end