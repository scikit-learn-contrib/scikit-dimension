function id = idtle(X,dists,epsilon)
% X - matrix of nearest neighbors (k x d), sorted by distance
% dists - nearest-neighbor distances (1 x k), sorted
    if nargin < 2
        error('Two parameters required: X - matrix of nearest neighbors (k x d), sorted by distance; dists - nearest-neighbor distances (1 x k), sorted');
    end
    if nargin == 2
        epsilon = 0.0001;
    end
    r = dists(end); % distance to k-th neighbor
    %% Boundary case 1: If $r = 0$, this is fatal, since the neighborhood would be degenerate.
    if r == 0
        error('All k-NN distances are zero!');
    end
    %% Main computation
    k = length(dists);
    V = squareform(pdist(X));
    Di = repmat(dists',1,k);
    Dj = Di';
    Z2 = 2*Di.^2 + 2*Dj.^2 - V.^2;
    S = r * (((Di.^2 + V.^2 - Dj.^2).^2 + 4*V.^2 .* (r^2 - Di.^2)).^0.5 - (Di.^2 + V.^2 - Dj.^2)) ./ (2*(r^2 - Di.^2));
    T = r * (((Di.^2 + Z2   - Dj.^2).^2 + 4*Z2   .* (r^2 - Di.^2)).^0.5 - (Di.^2 + Z2   - Dj.^2)) ./ (2*(r^2 - Di.^2));
    Dr = dists == r; % handle case of repeating k-NN distances
    S(Dr,:) = r * V(Dr,:).^2 ./ (r^2 + V(Dr,:).^2 - Dj(Dr,:).^2);
    T(Dr,:) = r * Z2(Dr,:)   ./ (r^2 + Z2(Dr,:)   - Dj(Dr,:).^2);
    %% Boundary case 2: If $u_i = 0$, then for all $1\leq j\leq k$ the measurements $s_{ij}$ and $t_{ij}$ reduce to $u_j$.
    Di0 = Di == 0; 
    T(Di0) = Dj(Di0);
    S(Di0) = Dj(Di0);
    %% Boundary case 3: If $u_j = 0$, then for all $1\leq j\leq k$ the measurements $s_{ij}$ and $t_{ij}$ reduce to $\frac{r v_{ij}}{r + v_{ij}}$.
    Dj0 = Dj == 0; 
    T(Dj0) = r * V(Dj0) ./ (r + V(Dj0));
    S(Dj0) = r * V(Dj0) ./ (r + V(Dj0));
    %% Boundary case 4: If $v_{ij} = 0$, then the measurement $s_{ij}$ is zero and must be dropped. The measurement $t_{ij}$ should be dropped as well.
    V0 = V == 0;
    V0(logical(eye(k))) = 0;
    T(V0) = r; % by setting to r, $t_{ij}$ will not contribute to the sum s1t
    S(V0) = r; % by setting to r, $s_{ij}$ will not contribute to the sum s1s
    nV0 = sum(V0(:)); % will subtract twice this number during ID computation below
    %% Drop T & S measurements below epsilon (V4: If $s_{ij}$ is thrown out, then for the sake of balance, $t_{ij}$ should be thrown out as well (or vice versa).)
    TSeps = T < epsilon | S < epsilon;
    TSeps(logical(eye(k))) = 0;
    nTSeps = sum(TSeps(:));
    T(TSeps) = r;
    T = log(T/r);
    S(TSeps) = r;
    S = log(S/r);
    T(logical(eye(k))) = 0; % delete diagonal elements
    S(logical(eye(k))) = 0;
    %% Sum over the whole matrices
    s1t = sum(T(:));
    s1s = sum(S(:));
    %% Drop distances below epsilon and compute sum
    Deps = dists < epsilon;
    nDeps = sum(Deps);
    dists = dists(nDeps+1:end);
    s2 = sum(log(dists/r));
    %% Compute ID, subtracting numbers of dropped measurements
    id = -2*(k^2-nTSeps-nDeps-nV0) / (s1t+s1s+2*s2);
end
