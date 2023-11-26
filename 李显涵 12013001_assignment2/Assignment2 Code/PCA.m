function [score] = PCA(images)
%PCA 
%去中心化
meas = images;
for i=1:size(meas,1)
    data = meas(i,:);
    [data,~] = mapminmax(data,0,1);
    meas(i,:) = data;
end
M_matrix = repmat(mean(meas,2),1,size(images,2));
meas = meas - M_matrix;
meas = meas.';

% 计算协方差矩阵
covar = cov(meas);
% 得到特征向量
[v,d] = eigs(covar);
% get the projection scores on the first two leading vectors.
% score = meas * v(:,1:2);

% get the projection scores on the first two leading vectors.
maxs = max(abs(d));
[a_1, b_1] = max(maxs);
maxs(b_1) = 0;
[a_2, b_2] = max(maxs);
p_1 = v(:,b_1);
p_2 = v(:,b_2);
score(:,1) =  meas * p_1;
score(:,2) =  meas * p_2;
end