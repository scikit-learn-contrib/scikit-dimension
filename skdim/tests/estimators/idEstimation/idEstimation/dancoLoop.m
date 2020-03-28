function d = dancoLoop(data,k)

size_data = length(data(1,:));
[inds,dists] = KNN(data,round(size_data*0.1),true);
d = [];
for i = 1:length(inds(:,1))
    [d(i),kl,mu,tau] = DANCoFit(data(:,inds(i,:)),k,'inds',inds,'dists',dists);
end
end
