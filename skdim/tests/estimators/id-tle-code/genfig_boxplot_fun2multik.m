function genfig_boxplot_fun2multik(dataSet,ks)

maxk = 200; % Largest k used for precomputed kNN distances

tic;

rng(42, 'twister'); % Random seed (42 used in SDM'19 paper)

theta = 0.975; % Ratio of variance to preserve by PCA (0.975 used in SDM'19 paper)

fprintf('\nloading data set: %s',dataSet);
X = dlmread(['data/real/' dataSet '.data'],' ');
[n,~] = size(X);

id_mle = zeros(n,1);
id_tle = zeros(n,1);
id_lcd = zeros(n,1);
id_mom = zeros(n,1);
id_ed = zeros(n,1);
id_ged = zeros(n,1);
id_lpca = zeros(n,1);

% Load or compute kNN distances
nnFilePrefix = ['data/real/knn/' dataSet '-k' num2str(maxk)];
nnidxFileNameMAT = [nnFilePrefix '-nnidx.mat'];
nndistsFileNameMAT = [nnFilePrefix '-nndists.mat'];
nnidxFileNameCSV = [nnFilePrefix '-nnidx.csv'];
nndistsFileNameCSV = [nnFilePrefix '-nndists.csv'];
if exist(nnidxFileNameMAT,'file') && exist(nndistsFileNameMAT,'file')
    % STRONGLY prefer .mat format since some methods are sensitive to rounding of distances
    fprintf('\nloading %d-nearest neighbors from .mat file...',maxk);
    load(nnidxFileNameMAT,'idx');
    load(nndistsFileNameMAT,'dists');
    idxmax = idx; %#ok<NODEF>
    distsmax = dists; %#ok<NODEF>
elseif exist(nnidxFileNameCSV,'file') && exist(nndistsFileNameCSV,'file')
    fprintf('\nloading %d-nearest neighbors from .csv file...',maxk);
    idxmax = csvread(nnidxFileNameCSV);
    distsmax = csvread(nndistsFileNameCSV);
else
    fprintf('\ncomputing %d-nearest neighbors...',max(ks));
    [idxmax,distsmax] = knnsearch(X,X,'K',max(ks)+1);
    idxmax = idxmax(:,2:end); % 2:end skips first neighbor - the point itself
    distsmax = distsmax(:,2:end);
end

warning('off'); % Because of PCA complaining about singular matrices

for k = ks

    idx = idxmax(:,1:k);
    dists = distsmax(:,1:k);

    fprintf('\n\nk = %d\nquery point:',k);
    for i = 1:n
        if mod(i,1000)==0, fprintf('\n%d',i); end
        KNN = X(idx(i,:),:);
        id_tle(i) = idtle(KNN,dists(i,:));
        id_lcd(i) = idlcd(KNN);
        id_mle(i) = idmle(dists(i,:)');
        id_mom(i) = idmom(dists(i,:)');
        id_ed(i) = ided(dists(i,:)');
        id_ged(i) = idged(dists(i,:)');
        id_lpca(i) = idlpca(KNN,theta);
    end
    
    csvwrite(['data/real/id/' dataSet '-k' num2str(k) '-id_mle.csv'],id_mle);
    csvwrite(['data/real/id/' dataSet '-k' num2str(k) '-id_tle.csv'],id_tle);
    csvwrite(['data/real/id/' dataSet '-k' num2str(k) '-id_lcd.csv'],id_lcd);
    csvwrite(['data/real/id/' dataSet '-k' num2str(k) '-id_mom.csv'],id_mom);
    csvwrite(['data/real/id/' dataSet '-k' num2str(k) '-id_ed.csv'],id_ed);
    csvwrite(['data/real/id/' dataSet '-k' num2str(k) '-id_ged.csv'],id_ged);
    csvwrite(['data/real/id/' dataSet '-k' num2str(k) '-id_lpca.csv'],id_lpca);
    
    h = figure;
    boxplot([id_mle id_tle id_lcd id_mom id_ed id_ged id_lpca],'Labels',{'MLE','TLE','LCD','MoM','ED','GED','LPCA'},'Whisker',1.5);
    ylabel('ID');
    title([dataSet ', k = ' num2str(k)]);
    saveas(h,['data/real/fig/' 'boxplot-' dataSet '-k' num2str(k) '.fig']);
    saveas(h,['data/real/fig/' 'boxplot-' dataSet '-k' num2str(k) '.png']);
    close(h);

end

warning('on');

fprintf('\n\n');

toc;