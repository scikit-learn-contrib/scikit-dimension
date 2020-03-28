rng(42, 'twister'); % Random seed (42 used in SDM'19 paper)

dataDist = 'Gaussian'; % Data distribution: 'Gaussian', 'Uniform' or 'Torus'

n = 10000; % No. of points in data set (10000 used in SDM'19 paper)
q = 1000;  % No. of points in query set (1000 used in SDM'19 paper)
d = 10;    % Dimensionality (10 used in SDM'19 paper)

ks = 10:10:100; % Values of k (10:10:100 used in SDM'19 paper)

runs = 10; % No. of runs (20 used in SDM'19 paper)

theta = 0.975; % Ratio of variance to preserve by PCA (0.975 used in SDM'19 paper)

numks = length(ks);

mlemean = zeros(1,numks);
mlevar = zeros(1,numks);
tlemean = zeros(1,numks);
tlevar = zeros(1,numks);
lcdmean = zeros(1,numks);
lcdvar = zeros(1,numks);
mommean = zeros(1,numks);
momvar = zeros(1,numks);
edmean = zeros(1,numks);
edvar = zeros(1,numks);
gedmean = zeros(1,numks);
gedvar = zeros(1,numks);
lpcamean = zeros(1,numks);
lpcavar = zeros(1,numks);

for r = 1:runs

    fprintf('\nrun = %d, k =',r);
    
    if strcmp(dataDist,'Gaussian')
        X = randn(n,d);
        Q = randn(q,d);
    elseif strcmp(dataDist,'Uniform')
        X = rand(n,d)-0.5;
        Q = rand(q,d)-0.5;
    elseif strcmp(dataDist,'Torus')
        X = rand(n,d);
        Q = rand(q,d);
    else
        error(['Unsupported data distribution: ' dataDist]);
    end
        
    if strcmp(dataDist,'Torus')
        [idxmax,distsmax] = knnsearch(X,Q,'K',max(ks),'Distance',@torusL2DistForKNNSearch);
    else
        [idxmax,distsmax] = knnsearch(X,Q,'K',max(ks));
    end
    
    for j = 1:numks
        
        k = ks(j);
        fprintf(' %d',k);
        
        id_mle = zeros(q,1);
        id_tle = zeros(q,1);
        id_lcd = zeros(q,1);
        id_mom = zeros(q,1);
        id_ed = zeros(q,1);
        id_ged = zeros(q,1);
        id_lpca = zeros(q,1);
        
        idx = idxmax(:,1:k);
        dists = distsmax(:,1:k);
        
        for i = 1:q
            KNN = X(idx(i,:),:);
            if strcmp(dataDist,'Torus')
                for l = 1:k
                    for m = 1:d
                        diff = abs(Q(i,m)-KNN(l,m));
                        if 1-diff < diff
                            if Q(i,m) < 0.5
                                KNN(l,m) = KNN(l,m)-1;
                            else
                                KNN(l,m) = KNN(l,m)+1;
                            end
                        end
                    end
                end
            end
            id_mle(i) = idmle(dists(i,:)');
            id_tle(i) = idtle(KNN,dists(i,:));
            id_lcd(i) = idlcd(KNN);
            id_mom(i) = idmom(dists(i,:)');
            id_ed(i) = ided(dists(i,:)');
            id_ged(i) = idged(dists(i,:)');
            id_lpca(i) = idlpca(KNN,theta);
        end
        
        mlemean(j) = mlemean(j) + mean(id_mle);
        mlevar(j) = mlevar(j) + std(id_mle);
        tlemean(j) = tlemean(j) + mean(id_tle);
        tlevar(j) = tlevar(j) + std(id_tle);
        lcdmean(j) = lcdmean(j) + mean(id_lcd);
        lcdvar(j) = lcdvar(j) + std(id_lcd);
        mommean(j) = mommean(j) + mean(id_mom);
        momvar(j) = momvar(j) + std(id_mom);
        edmean(j) = edmean(j) + mean(id_ed);
        edvar(j) = edvar(j) + std(id_ed);
        gedmean(j) = gedmean(j) + mean(id_ged);
        gedvar(j) = gedvar(j) + std(id_ged);
        lpcamean(j) = lpcamean(j) + mean(id_lpca);
        lpcavar(j) = lpcavar(j) + std(id_lpca);
        
    end    
end

fprintf('\n\n');

mlemean = mlemean / runs;
mlevar = mlevar / runs;
tlemean = tlemean / runs;
tlevar = tlevar / runs;
lcdmean = lcdmean / runs;
lcdvar = lcdvar / runs;
mommean = mommean / runs;
momvar = momvar / runs;
edmean = edmean / runs;
edvar = edvar / runs;
gedmean = gedmean / runs;
gedvar = gedvar / runs;
lpcamean = lpcamean / runs;
lpcavar = lpcavar / runs;

figure;
hold on;
plot(ks,mlemean+mlevar,'b-.');
plot(ks,mlemean,'bo-');
plot(ks,mlemean-mlevar,'b-.');
plot(ks,tlemean+tlevar,'r-.');
plot(ks,tlemean,'r+-');
plot(ks,tlemean-tlevar,'r-.');
plot(ks,lcdmean+lcdvar,'m-.');
plot(ks,lcdmean,'mo-');
plot(ks,lcdmean-lcdvar,'m-.');
plot(ks,mommean+momvar,'g-.');
plot(ks,mommean,'g*-');
plot(ks,mommean-momvar,'g-.');
plot(ks,edmean+edvar,'k-.');
plot(ks,edmean,'k*-');
plot(ks,edmean-edvar,'k-.');
plot(ks,gedmean+gedvar,'c-.');
plot(ks,gedmean,'cx-');
plot(ks,gedmean-gedvar,'c-.');
plot(ks,lpcamean+lpcavar,'y-.');
plot(ks,lpcamean,'yx-');
plot(ks,lpcamean-lpcavar,'y-.');
plot(ks,repmat(d,1,numks),'k-');
hold off;
xlabel('k');
ylabel('ID');
title(['i.i.d. ' dataDist ', n = ' int2str(n) ', q = ' int2str(q) ', d = ' int2str(d) ', runs = ' int2str(runs)]);
legend('MLE: \mu+\sigma','MLE: \mu','MLE: \mu-\sigma',...
       'TLE: \mu+\sigma','TLE: \mu','TLE: \mu-\sigma',...
       'LCD: \mu+\sigma','LCD: \mu','LCD: \mu-\sigma',...
       'MoM: \mu+\sigma','MoM: \mu','MoM: \mu-\sigma',...
       'ED: \mu+\sigma','ED: \mu','ED: \mu-\sigma',...
       'GED: \mu+\sigma','GED: \mu','GED: \mu-\sigma',...
       'LPCA: \mu+\sigma','LPCA: \mu','LPCA: \mu-\sigma',...
       'The Truth','Location','NorthWest');
