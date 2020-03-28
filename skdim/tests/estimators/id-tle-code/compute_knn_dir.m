function compute_knn_dir(dataPath,k)

dataExt = '.data';
dataDlm = ' ';

listing = dir([dataPath '*' dataExt]);

for i = 1:length(listing)

    tic;

    dataSet = listing(i).name(1:end-length(dataExt));
    fprintf('\nloading data set: %s',dataSet);
    X = dlmread([dataPath dataSet dataExt],dataDlm);
    
    fprintf('\ncomputing %d-nearest neighbors...',k);
    [idx,dists] = knnsearch(X,X,'K',k+1);
    idx = idx(:,2:end); %#ok<NASGU> % 2:end skips first neighbor - the point itself
    dists = dists(:,2:end); %#ok<NASGU>
    
    fprintf('\nwriting output...');
    % csvwrite([dataPath 'knn/' dataSet '-k' num2str(k) '-nnidx.csv'],idx);
    % csvwrite([dataPath 'knn/' dataSet '-k' num2str(k) '-nndists.csv'],dists);
    save([dataPath 'knn/' dataSet '-k' num2str(k) '-nnidx.mat'],'idx');
    save([dataPath 'knn/' dataSet '-k' num2str(k) '-nndists.mat'],'dists');
    
    fprintf('\n');

    toc;
    
end
