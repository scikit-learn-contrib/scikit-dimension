X=randsphere(3,1000,1)';
% Initialize some variables
gamma = 1;
M = 1; N = 10;
samp_points = size(X, 1) - 10:size(X, 1) - 1;
k = 6;
Q = length(samp_points);
knnlenavg_vec = zeros(M, Q);
knnlenstd_vec = zeros(M, Q);
dvec = zeros(M, 1);

% Compute Euclidean distance matrix
D = find_nn(X, k * 8); % wide range to deal with permutations

% Make M estimates
for i=1:M

    % Perform resampling estimation of mean k-nn length
    j = 1;
    for n=samp_points

        % Sum cumulative distances over N random permutations
        knnlen1 = 0;
        knnlen2 = 0;
        for trial=1:N

            % Construct random permutation of data (throws out
            % some points)
            indices = randperm(size(X, 1));
            indices = indices(1:n);
            Dr = D(indices,:);
            Drr = Dr(:,indices);

            % Compute sum of distances to k nearest neighbors
            L = 0;
            Drr = sort(Drr, 1);
            for l=1:size(Drr, 2)
                ind = min(find(Drr(:,l) ~= 0));
                L = L + sum(Drr(ind + 1:min([ind + k size(Drr, 2)]), l));
            end

            % Accumulate sum and squared sum over all trials
            knnlen1 = knnlen1 + L;
            knnlen2 = knnlen2 + L^2;
        end

        % Compute average and standard deviation over N trials
        knnlenavg_vec(i, j) = knnlen1 / N;
        knnlenstd_vec(i, j) = sqrt((knnlen2 - (knnlen1 / N) ^ 2 * N) / (N - 1));

        % Update counter
        j = j + 1;
    end

    % Compute least squares estimate of intrinsic dimensionality
    A = [log(samp_points)' ones(Q,1)];
    sol1 = inv(A' * A) * A' * log(knnlenavg_vec(i,:))';
    
    dvec(i) = gamma / (1 - sol1(1));
end

% Average over all M estimates
no_dims = mean(abs(dvec));   