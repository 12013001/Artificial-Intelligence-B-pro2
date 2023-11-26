function [Z,c] = HC(score,labels)
% Z = linkage(score(:,1:2),'single','euclidean');
% Z = linkage(score(:,1:2),'complete','euclidean');
Z = linkage(score(:,1:2),'average','euclidean');
%create the linkage tree using average-link
c = cluster(Z,'maxclust',3);
% See how the cluster assignments correspond to the three species.
crosstab(c, labels);
end

