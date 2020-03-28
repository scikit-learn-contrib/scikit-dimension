function id = idmind_mli(dists,D)
    k = size(dists,1);
    n = size(dists,2);
    nlogk = n*log(k);
    rho = dists(1,:)./dists(k,:);
    ll = zeros(1,D);
    sum1 = sum(log(rho));
    for d = 1:D
        sum2 = sum(log(1-rho.^d));
        ll(d) = nlogk + n*log(d) + (d-1)*sum1 + (k-1)*sum2;
    end
    [~,id] = max(ll);
end
