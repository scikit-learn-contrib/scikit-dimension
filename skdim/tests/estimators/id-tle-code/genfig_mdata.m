dataPath = 'data/m10000/';
dataSet = 'm1'; % m1, m2, ..., m9, m10a, m10b, m10c, m11, m12, m13
n = 10000;
runs = 20; % No. of runs (20 used in SDM'19 paper)
ks = [20 50 100]; % Values of k ([20 50 100] used in SDM'19 paper)

id_mle = zeros(n,runs);
id_tle = zeros(n,runs);
id_lcd = zeros(n,runs);
id_mom = zeros(n,runs);
id_ed = zeros(n,runs);
id_ged = zeros(n,runs);
id_lpca = zeros(n,runs);

numks = length(ks);

id_mle_all = zeros(n*runs,numks);
id_tle_all = zeros(n*runs,numks);
id_lcd_all = zeros(n*runs,numks);
id_mom_all = zeros(n*runs,numks);
id_ed_all = zeros(n*runs,numks);
id_ged_all = zeros(n*runs,numks);
id_lpca_all = zeros(n*runs,numks);

for i = 1:numks
    k = ks(i);
    for r = 1:runs
        rstr = num2str(r-1);
        if r-1 < 10, rstr = ['0' rstr]; end %#ok<AGROW>
        namePrefix = [dataPath 'id/' dataSet '-' rstr '-k' num2str(k)];
        id_mle(:,r) = csvread([namePrefix '-id_mle.csv']);
        id_tle(:,r) = csvread([namePrefix '-id_tle.csv']);
        id_lcd(:,r) = csvread([namePrefix '-id_lcd.csv']);
        id_mom(:,r) = csvread([namePrefix '-id_mom.csv']);
        id_ed(:,r) = csvread([namePrefix '-id_ed.csv']);
        id_ged(:,r) = csvread([namePrefix '-id_ged.csv']);
        id_lpca(:,r) = csvread([namePrefix '-id_lpca.csv']);
    end
    id_mle_all(:,i) = id_mle(:);
    id_tle_all(:,i) = id_tle(:);
    id_lcd_all(:,i) = id_lcd(:);
    id_mom_all(:,i) = id_mom(:);
    id_ed_all(:,i) = id_ed(:);
    id_ged_all(:,i) = id_ged(:);
    id_lpca_all(:,i) = id_lpca(:);
end

for i = 1:numks
    k = ks(i);
    h = figure;
    boxplot([id_mle_all(:,i) id_tle_all(:,i) id_lcd_all(:,i) id_mom_all(:,i) id_ed_all(:,i) id_ged_all(:,i) id_lpca_all(:,i)],'Labels',{'MLE','TLE','LCD','MoM','ED','GED','LPCA'},'Whisker',1.5);
    ylabel('ID');
    title([dataSet ', k = ' num2str(k)]);
    saveas(h,[dataPath 'fig/' 'boxplot-' dataSet '-k' num2str(k) '.fig']);
    saveas(h,[dataPath 'fig/' 'boxplot-' dataSet '-k' num2str(k) '.eps']);
    close(h);
end
