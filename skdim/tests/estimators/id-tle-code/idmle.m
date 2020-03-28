function id = idmle(dists)
% dists - nearest-neighbor distances (k x 1, or k x n), sorted
    k = size(dists,1);
    w = repmat(dists(k,:),k,1);
    xi = sum(log(dists ./ w)) / (k-1);
    id = -(1 ./ xi);
end