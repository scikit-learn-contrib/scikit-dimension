size_data = length(data(:,1));
[inds,dists] = KNN(data',round(size_data*0.1),true);
d = [];
for i = 1:size_data
	  d(i) = intrinsic_dim(data(inds(i,:),:),'GMST');
end
