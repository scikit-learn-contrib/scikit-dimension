function compute_ids_dir(dataPath,ks)

tic;

rng(42, 'twister');

theta = 0.975; % ratio of variance to preserve by PCA

dataExt = '.data';
dataDlm = ' ';

maxk = max(ks);

listing = dir([dataPath '*' dataExt]);

for f = 1:length(listing)

    dataSet = listing(f).name(1:end-length(dataExt));
    fprintf('\nloading data set: %s',dataSet);
    X = dlmread([dataPath dataSet dataExt],dataDlm);
    
    n = size(X,1);
    
    id_mle = ones(n,1);
    id_tle = zeros(n,1);
    id_lcd = zeros(n,1);
    id_mom = zeros(n,1);
    id_ed = zeros(n,1);
    id_ged = zeros(n,1);
    id_lpca = zeros(n,1);

    fprintf('\ncomputing %d-nearest neighbors...',maxk);
    [idxmax,distsmax] = knnsearch(X,X,'K',maxk+1);
    idxmax = idxmax(:,2:end); % 2:end skips first neighbor - the point itself
    distsmax = distsmax(:,2:end);
    
    warning('off'); % because of PCA and singular matrices
    
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
        
        csvwrite([dataPath 'id/' dataSet '-k' num2str(k) '-id_mle.csv'],id_mle);
        csvwrite([dataPath 'id/' dataSet '-k' num2str(k) '-id_tle.csv'],id_tle);
        csvwrite([dataPath 'id/' dataSet '-k' num2str(k) '-id_lcd.csv'],id_lcd);
        csvwrite([dataPath 'id/' dataSet '-k' num2str(k) '-id_mom.csv'],id_mom);
        csvwrite([dataPath 'id/' dataSet '-k' num2str(k) '-id_ed.csv'],id_ed);
        csvwrite([dataPath 'id/' dataSet '-k' num2str(k) '-id_ged.csv'],id_ged);
        csvwrite([dataPath 'id/' dataSet '-k' num2str(k) '-id_lpca.csv'],id_lpca);
        
    end
    
    warning('on');
    
    fprintf('\n');

end

toc;