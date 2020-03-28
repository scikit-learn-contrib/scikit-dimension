function id = idlcd(X)
% X - matrix of nearest neighbors (k x d)

    tmp = GetDim(X'); % Hein's implementation
    id = tmp(2);
end