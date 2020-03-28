function id = idgpca(X,theta)
% X - data set (n x d)
% theta - ratio of variance to preserve (theta \in [0,1])
    [~,~,~,~,explained] = pca(X);
    d = size(X,2);
    theta = theta*100;
    id = 1;
    sumexp = explained(id);
    while id < d && sumexp < theta
        id = id+1;
        sumexp = sumexp + explained(id);
    end
end
