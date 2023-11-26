function [idx,C] = k_means(score)
[idx,C] = kmeans(score, 3);
x1 = min(score(:, 1)):0.01:max(score(:, 1));
x2 = min(score(:, 2)):0.01:max(score(:, 2));
[x1G,x2G] = meshgrid(x1,x2);
XGrid = [x1G(:),x2G(:)]; % Defines a fine grid on the plot
idx2Region = kmeans(XGrid,3,'MaxIter',50,'Start',C);
end

