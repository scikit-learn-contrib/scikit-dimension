function id = idged(dists)
% dists - nearest-neighbor distances (k x 1, or k x n), sorted
    k = size(dists,1);
    n = size(dists,2);
    B = zeros(k,n);
    M = zeros(k,n);
    for i = 1:k
        for j = 1:k
            B(j,:) = (log2(i/j) ./ (log2(dists(i,:)./dists(j,:))));
        end
        B = sort(B);
        midIdx = floor((sum(B~=Inf)+1)/2);
        for j = 1:n
            M(i,j) = B(midIdx(j),j);
        end
    end
    M = sort(M);
    id = M(floor((k+1)/2),:);
end