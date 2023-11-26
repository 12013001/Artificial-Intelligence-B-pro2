clc;
clear;
close all;

% Load dataset in Matlab
load mnist-1-5-8.mat;

% % To visualise one image , you can simply try:
% i = unidrnd(600);%随机给我挑一个
% im = reshape(images(:, i), [28, 28]);
% figure(1);
% imshow(im,[]); 

%----------------------------------------PCA
score = PCA(images);

l1 = find(labels==1);
l5 = find(labels==5);
l8 = find(labels==8);

figure(2);
hold on;
plot(score(l1, 1), score(l1, 2), 'o');
plot(score(l5, 1), score(l5, 2), '*');
plot(score(l8, 1), score(l8, 2), '+');
grid on;
title('PCA');
legend('digit"1"', 'digit"5"', 'digit"8"');

% -----------------hierarchical clustering
[Z,c] = HC(score,labels);

% dendrogram plot
figure(3);
dendrogram(Z);
title('Dendrogram Plot')

figure(4);
hold on;
gscatter(score(:,1), score(:,2), c);
grid on;
title('Hierarchical Clustering');
legend('Cluster 1', 'Cluster 2', 'Cluster 3');

% ----------------------------k-means
[idx,C] = k_means(score);

figure(5);
hold on;
gscatter(score(:,1), score(:,2), idx);
plot(C(:,1), C(:,2), 'k*')
grid on;
title('K-Means')
legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster Centroid');

% --------------------------------------------GMM
figure(6);
k = 3; % Number of GMM components
options = statset('MaxIter',1000);

Sigma = {'diagonal','full'}; % Options for covariance matrix type
nSigma = numel(Sigma);

SharedCovariance = {true,false}; % Indicator for identical or nonidentical covariance matrices
SCtext = {'true','false'};
nSC = numel(SharedCovariance);

d = 500; % Grid length
x1 = linspace(min(score(:,1))-2, max(score(:,1))+2, d);
x2 = linspace(min(score(:,2))-2, max(score(:,2))+2, d);
[x1grid,x2grid] = meshgrid(x1,x2);
X0 = [x1grid(:) x2grid(:)];

threshold = sqrt(chi2inv(0.99,2));
count = 1;
for i = 1:nSigma
    for j = 1:nSC
        gmfit = fitgmdist(score,k,'CovarianceType',Sigma{i}, ...
            'SharedCovariance',SharedCovariance{j},'Options',options); % Fitted GMM
        clusterX = cluster(gmfit,score); % Cluster index 
        mahalDist = mahal(gmfit,X0); % Distance from each grid point to each GMM component
        % Draw ellipsoids over each GMM component and show clustering result.
        subplot(2,2,count);
        h1 = gscatter(score(:,1),score(:,2),clusterX);
        hold on
            for m = 1:k
                idx = mahalDist(:,m)<=threshold;
                Color = h1(m).Color*0.75 - 0.5*(h1(m).Color - 1);
                h2 = plot(X0(idx,1),X0(idx,2),'.','Color',Color,'MarkerSize',1);
                uistack(h2,'bottom');
            end    
        plot(gmfit.mu(:,1),gmfit.mu(:,2),'kx','LineWidth',2,'MarkerSize',10)
        title(sprintf('Sigma is %s\nSharedCovariance = %s',Sigma{i},SCtext{j}),'FontSize',8)
        legend(h1,{'1','2','3'})
        hold off
        count = count + 1;
    end
end