function id = idlpca(X,theta)
% X - matrix of nearest neighbors (k x d)
% theta - ratio of variance to preserve (theta \in [0,1])
    if nargin < 2
        error('Two parameters required: X - matrix of nearest neighbors (k x d); theta - ratio of variance to preserve');
    end
    id = idgpca(X,theta);
end
