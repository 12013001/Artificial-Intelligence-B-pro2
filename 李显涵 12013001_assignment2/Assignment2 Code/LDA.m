function [score] = LDA(images,labels)
images = images.';

class1 = images(labels==1,:);
class5 = images(labels==5,:);
class8 = images(labels==8,:);

% class means
m1 = mean(class1);
m5 = mean(class5);
m8 = mean(class8);
m_images = mean(images);

% class covariance matrix
s1 = cov(class1);
s5 = cov(class5);
s8 = cov(class8);

% within class scatter matrix
sw = s1 + s5 + s8;

% between class scatter matrix
mb = zeros(3, 784);
mb(1, :) =  m1 - m_images;
mb(2, :) =  m5 - m_images;
mb(3, :) =  m8 - m_images;
sb = mb.' * mb;

% computing the LDA projection vector
[v, d] = eigs((inv(sw + 1e-10 * eye(784))) * sb);

% computing the projection score:
% score = images * v(:, 1:2);

maxs = max(abs(d));
[a_1, b_1] = max(maxs);
maxs(b_1) = 0;
[a_2, b_2] = max(maxs);
p_1 = v(:,b_1);
p_2 = v(:,b_2);
score(:,1) =  images * p_1;
score(:,2) =  images * p_2;
end