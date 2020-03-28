rng(42, 'twister'); % Random seed (42 used in SDM'19 paper)

dataDist = 'Gaussian'; % Data distribution: 'Gaussian', 'Uniform' or 'Torus'

% Ratios of outliers/inliers to consider in chart
% (works only with 'Gaussian' and 'Uniform' data)
outlierRatio = 0.1; % (0.1 used in SDM'19 paper)
inlierRatio = 0.1;  % (0.1 used in SDM'19 paper)

n = 10000; % No. of points in data set (10000 used in SDM'19 paper)
q = 1000;  % No. of points in query set (1000 used in SDM'19 paper)
k = 20;    % No. of neighbors (20 used in SDM'19 paper)

ds = 2:1:20; % Dimensionalities (2:1:20 used in SDM'19 paper)

runs = 20; % No. of runs (20 used in SDM'19 paper)

theta = 0.975; % Ratio of variance to preserve by PCA (0.975 used in SDM'19 paper)

numds = length(ds);
numout = floor(q*outlierRatio);
numin = floor(q*inlierRatio);

corr_dim_val = zeros(1,numds); % one value for whole data set

mind_ml1_val = zeros(1,numds); % one value for whole query set
mind_mli_val = zeros(1,numds); % one value for whole query set

mlemean = zeros(1,numds);
mlevar = zeros(1,numds);
tlemean = zeros(1,numds);
tlevar = zeros(1,numds);
lcdmean = zeros(1,numds);
lcdvar = zeros(1,numds);
mommean = zeros(1,numds);
momvar = zeros(1,numds);
edmean = zeros(1,numds);
edvar = zeros(1,numds);
gedmean = zeros(1,numds);
gedvar = zeros(1,numds);
lpcamean = zeros(1,numds);
lpcavar = zeros(1,numds);

% For outliers
omind_ml1_val = zeros(1,numds); % one value for whole query set
omind_mli_val = zeros(1,numds); % one value for whole query set

omlemean = zeros(1,numds);
omlevar = zeros(1,numds);
otlemean = zeros(1,numds);
otlevar = zeros(1,numds);
olcdmean = zeros(1,numds);
olcdvar = zeros(1,numds);
omommean = zeros(1,numds);
omomvar = zeros(1,numds);
oedmean = zeros(1,numds);
oedvar = zeros(1,numds);
ogedmean = zeros(1,numds);
ogedvar = zeros(1,numds);
olpcamean = zeros(1,numds);
olpcavar = zeros(1,numds);

% For inliers
imind_ml1_val = zeros(1,numds); % one value for whole query set
imind_mli_val = zeros(1,numds); % one value for whole query set

imlemean = zeros(1,numds);
imlevar = zeros(1,numds);
itlemean = zeros(1,numds);
itlevar = zeros(1,numds);
ilcdmean = zeros(1,numds);
ilcdvar = zeros(1,numds);
imommean = zeros(1,numds);
imomvar = zeros(1,numds);
iedmean = zeros(1,numds);
iedvar = zeros(1,numds);
igedmean = zeros(1,numds);
igedvar = zeros(1,numds);
ilpcamean = zeros(1,numds);
ilpcavar = zeros(1,numds);

for r = 1:runs

    fprintf('\nrun = %d, d =',r);
    
    for j = 1:numds
        
        d = ds(j);
        fprintf(' %d',d);
        
        if strcmp(dataDist,'Gaussian')
            X = randn(n,d);
            Q = randn(q,d);
        elseif strcmp(dataDist,'Uniform')
            X = rand(n,d)-0.5;
            Q = rand(q,d)-0.5; % centering so that norm is outlier score
        elseif strcmp(dataDist,'Torus')
            X = rand(n,d);
            Q = rand(q,d);
        else
            error(['Unsupported data distribution: ' dataDist]);
        end
        
        if strcmp(dataDist,'Torus')
            [idx,dists] = knnsearch(X,Q,'K',k,'Distance',@torusL2DistForKNNSearch);
        else
            [idx,dists] = knnsearch(X,Q,'K',k);
        end
        
        normsQ = sum(Q.^2,2);
        [~,idxQ] = sort(normsQ,'descend');
        idxQout = idxQ(1:numout);
        idxQin = idxQ(end-numin+1:end);
        
        id_tle = zeros(q,1);
        id_lcd = zeros(q,1);
        id_mle = zeros(q,1);
        id_mom = zeros(q,1);
        id_ed = zeros(q,1);
        id_ged = zeros(q,1);
        id_lpca = zeros(q,1);
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
            id_tle(i) = idtle(KNN,dists(i,:));
            id_lcd(i) = idlcd(KNN);
            id_mle(i) = idmle(dists(i,:)');
            id_mom(i) = idmom(dists(i,:)');
            id_ed(i) = ided(dists(i,:)');
            id_ged(i) = idged(dists(i,:)');
            id_lpca(i) = idlpca(KNN,theta);
       end
       mind_ml1_val(j) = mind_ml1_val(j) + idmind_ml1(dists');
       mind_mli_val(j) = mind_mli_val(j) + idmind_mli(dists',d);
       omind_ml1_val(j) = omind_ml1_val(j) + idmind_ml1(dists(idxQout,:)');
       omind_mli_val(j) = omind_mli_val(j) + idmind_mli(dists(idxQout,:)',d);
       imind_ml1_val(j) = imind_ml1_val(j) + idmind_ml1(dists(idxQin,:)');
       imind_mli_val(j) = imind_mli_val(j) + idmind_mli(dists(idxQin,:)',d);
       
       tmp = GetDim(X'); % Hein's implementation
       corr_dim_val(j) = corr_dim_val(j) + tmp(2);
       
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
       
       omlemean(j) = omlemean(j) + mean(id_mle(idxQout));
       omlevar(j) = omlevar(j) + std(id_mle(idxQout));
       otlemean(j) = otlemean(j) + mean(id_tle(idxQout));
       otlevar(j) = otlevar(j) + std(id_tle(idxQout));
       olcdmean(j) = olcdmean(j) + mean(id_lcd(idxQout));
       olcdvar(j) = olcdvar(j) + std(id_lcd(idxQout));
       omommean(j) = omommean(j) + mean(id_mom(idxQout));
       omomvar(j) = omomvar(j) + std(id_mom(idxQout));
       oedmean(j) = oedmean(j) + mean(id_ed(idxQout));
       oedvar(j) = oedvar(j) + std(id_ed(idxQout));
       ogedmean(j) = ogedmean(j) + mean(id_ged(idxQout));
       ogedvar(j) = ogedvar(j) + std(id_ged(idxQout));
       olpcamean(j) = olpcamean(j) + mean(id_lpca(idxQout));
       olpcavar(j) = olpcavar(j) + std(id_lpca(idxQout));
       
       imlemean(j) = imlemean(j) + mean(id_mle(idxQin));
       imlevar(j) = imlevar(j) + std(id_mle(idxQin));
       itlemean(j) = itlemean(j) + mean(id_tle(idxQin));
       itlevar(j) = itlevar(j) + std(id_tle(idxQin));
       ilcdmean(j) = ilcdmean(j) + mean(id_lcd(idxQin));
       ilcdvar(j) = ilcdvar(j) + std(id_lcd(idxQin));
       imommean(j) = imommean(j) + mean(id_mom(idxQin));
       imomvar(j) = imomvar(j) + std(id_mom(idxQin));
       iedmean(j) = iedmean(j) + mean(id_ed(idxQin));
       iedvar(j) = iedvar(j) + std(id_ed(idxQin));
       igedmean(j) = igedmean(j) + mean(id_ged(idxQin));
       igedvar(j) = igedvar(j) + std(id_ged(idxQin));
       ilpcamean(j) = ilpcamean(j) + mean(id_lpca(idxQin));
       ilpcavar(j) = ilpcavar(j) + std(id_lpca(idxQin));

    end
end
fprintf('\n\n');

mind_ml1_val = mind_ml1_val / runs;
mind_mli_val = mind_mli_val / runs;
omind_ml1_val = omind_ml1_val / runs;
omind_mli_val = omind_mli_val / runs;
imind_ml1_val = imind_ml1_val / runs;
imind_mli_val = imind_mli_val / runs;

corr_dim_val = corr_dim_val / runs;

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

omlemean = omlemean / runs;
omlevar = omlevar / runs;
otlemean = otlemean / runs;
otlevar = otlevar / runs;
olcdmean = olcdmean / runs;
olcdvar = olcdvar / runs;
omommean = omommean / runs;
omomvar = omomvar / runs;
oedmean = oedmean / runs;
oedvar = oedvar / runs;
ogedmean = ogedmean / runs;
ogedvar = ogedvar / runs;
olpcamean = olpcamean / runs;
olpcavar = olpcavar / runs;

imlemean = imlemean / runs;
imlevar = imlevar / runs;
itlemean = itlemean / runs;
itlevar = itlevar / runs;
ilcdmean = ilcdmean / runs;
ilcdvar = ilcdvar / runs;
imommean = imommean / runs;
imomvar = imomvar / runs;
iedmean = iedmean / runs;
iedvar = iedvar / runs;
igedmean = igedmean / runs;
igedvar = igedvar / runs;
ilpcamean = ilpcamean / runs;
ilpcavar = ilpcavar / runs;

figure;
hold on;
plot(ds,mlemean+mlevar,'b-.');
plot(ds,mlemean,'bo-');
plot(ds,mlemean-mlevar,'b-.');
plot(ds,tlemean+tlevar,'r-.');
plot(ds,tlemean,'r+-');
plot(ds,tlemean-tlevar,'r-.');
plot(ds,lcdmean+lcdvar,'m-.');
plot(ds,lcdmean,'mo-');
plot(ds,lcdmean-lcdvar,'m-.');
plot(ds,mommean+momvar,'g-.');
plot(ds,mommean,'g*-');
plot(ds,mommean-momvar,'g-.');
plot(ds,edmean+edvar,'k-.');
plot(ds,edmean,'k*-');
plot(ds,edmean-edvar,'k-.');
plot(ds,gedmean+gedvar,'c-.');
plot(ds,gedmean,'cx-');
plot(ds,gedmean-gedvar,'c-.');
plot(ds,lpcamean+lpcavar,'y-.');
plot(ds,lpcamean,'yx-');
plot(ds,lpcamean-lpcavar,'y-.');
plot(ds,mind_ml1_val,'m^-');
plot(ds,mind_mli_val,'mv-');
plot(ds,corr_dim_val,'mx-');
plot(ds,ds,'k-');
hold off;
xlabel('Dimensionality');
ylabel('ID');
title(['i.i.d. ' dataDist ', n = ' int2str(n) ', q = ' int2str(q) ', k = ' int2str(k) ', runs = ' int2str(runs)]);
legend('MLE: \mu+\sigma','MLE: \mu','MLE: \mu-\sigma',...
       'TLE: \mu+\sigma','TLE: \mu','TLE: \mu-\sigma',...
       'LCD: \mu+\sigma','LCD: \mu','LCD: \mu-\sigma',...
       'MoM: \mu+\sigma','MoM: \mu','MoM: \mu-\sigma',...
       'ED: \mu+\sigma','ED: \mu','ED: \mu-\sigma',...
       'GED: \mu+\sigma','GED: \mu','GED: \mu-\sigma',...
       'LPCA: \mu+\sigma','LPCA: \mu','LPCA: \mu-\sigma',...
       'MiND ml1','MiND mli','CorrDim','The Truth','Location','NorthWest');

figure;
hold on;
plot(ds,omlemean+omlevar,'b-.');
plot(ds,omlemean,'bo-');
plot(ds,omlemean-omlevar,'b-.');
plot(ds,otlemean+otlevar,'r-.');
plot(ds,otlemean,'r+-');
plot(ds,otlemean-otlevar,'r-.');
plot(ds,olcdmean+olcdvar,'m-.');
plot(ds,olcdmean,'mo-');
plot(ds,olcdmean-olcdvar,'m-.');
plot(ds,omommean+omomvar,'g-.');
plot(ds,omommean,'g*-');
plot(ds,omommean-omomvar,'g-.');
plot(ds,oedmean+oedvar,'k-.');
plot(ds,oedmean,'k*-');
plot(ds,oedmean-oedvar,'k-.');
plot(ds,ogedmean+ogedvar,'c-.');
plot(ds,ogedmean,'cx-');
plot(ds,ogedmean-ogedvar,'c-.');
plot(ds,olpcamean+olpcavar,'y-.');
plot(ds,olpcamean,'yx-');
plot(ds,olpcamean-olpcavar,'y-.');
plot(ds,omind_ml1_val,'m^-');
plot(ds,omind_mli_val,'mv-');
% plot(ds,corr_dim_val,'gx-');
plot(ds,ds,'k-');
hold off;
xlabel('Dimensionality');
ylabel('ID of outliers');
title(['i.i.d. ' dataDist ', n = ' int2str(n) ', q = ' int2str(q) ', k = ' int2str(k) ', runs = ' int2str(runs)]);
legend('MLE: \mu+\sigma','MLE: \mu','MLE: \mu-\sigma',...
       'TLE: \mu+\sigma','TLE: \mu','TLE: \mu-\sigma',...
       'LCD: \mu+\sigma','LCD: \mu','LCD: \mu-\sigma',...
       'MoM: \mu+\sigma','MoM: \mu','MoM: \mu-\sigma',...
       'ED: \mu+\sigma','ED: \mu','ED: \mu-\sigma',...
       'GED: \mu+\sigma','GED: \mu','GED: \mu-\sigma',...
       'LPCA: \mu+\sigma','LPCA: \mu','LPCA: \mu-\sigma',...
       'MiND ml1','MiND mli','The Truth','Location','NorthWest');

figure;
hold on;
plot(ds,imlemean+imlevar,'b-.');
plot(ds,imlemean,'bo-');
plot(ds,imlemean-imlevar,'b-.');
plot(ds,itlemean+itlevar,'r-.');
plot(ds,itlemean,'r+-');
plot(ds,itlemean-itlevar,'r-.');
plot(ds,ilcdmean+ilcdvar,'m-.');
plot(ds,ilcdmean,'mo-');
plot(ds,ilcdmean-ilcdvar,'m-.');
plot(ds,imommean+imomvar,'g-.');
plot(ds,imommean,'g*-');
plot(ds,imommean-imomvar,'g-.');
plot(ds,iedmean+iedvar,'k-.');
plot(ds,iedmean,'k*-');
plot(ds,iedmean-iedvar,'k-.');
plot(ds,igedmean+igedvar,'c-.');
plot(ds,igedmean,'cx-');
plot(ds,igedmean-igedvar,'c-.');
plot(ds,ilpcamean+ilpcavar,'y-.');
plot(ds,ilpcamean,'yx-');
plot(ds,ilpcamean-ilpcavar,'y-.');
plot(ds,imind_ml1_val,'m^-');
plot(ds,imind_mli_val,'mv-');
% plot(ds,corr_dim_val,'gx-');
plot(ds,ds,'k-');
hold off;
xlabel('Dimensionality');
ylabel('ID of inliers');
title(['i.i.d. ' dataDist ', n = ' int2str(n) ', q = ' int2str(q) ', k = ' int2str(k) ', runs = ' int2str(runs)]);
legend('MLE: \mu+\sigma','MLE: \mu','MLE: \mu-\sigma',...
       'TLE: \mu+\sigma','TLE: \mu','TLE: \mu-\sigma',...
       'LCD: \mu+\sigma','LCD: \mu','LCD: \mu-\sigma',...
       'MoM: \mu+\sigma','MoM: \mu','MoM: \mu-\sigma',...
       'ED: \mu+\sigma','ED: \mu','ED: \mu-\sigma',...
       'GED: \mu+\sigma','GED: \mu','GED: \mu-\sigma',...
       'LPCA: \mu+\sigma','lPCA: \mu','LPCA: \mu-\sigma',...
       'MiND ml1','MiND mli','The Truth','Location','NorthWest');
