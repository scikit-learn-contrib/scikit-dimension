function id = idmind_ml1(dists)
    n = size(dists,2);
    dists = dists(1:2,:).^2; % need only squared dists to first 2 neighbors
    s = sum(log(dists(1,:)./dists(2,:)));
    id = -2/(s/n);
end
