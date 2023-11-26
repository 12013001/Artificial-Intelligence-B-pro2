clc;
clear;
close all;

% Load dataset in Matlab
load mnist-1-5-8.mat;

% 将5的标签设置为1，其余的为0.
a = zeros(size(labels));
a(labels==5) = 1;

% split the data into 5 fold.
cvo = cvpartition(a,'k',5);

predict_label_total = [];
test_label_vector_total = [];

images = images';
%-----------------------------------5-fold cross validation setting
accuracy_avg = 0;
for i = 1:5
trIdx = cvo.training(i); %% get the index of training samples
teIdx = cvo.test(i); %% get the index of the test samples
training_label_vector = a(trIdx); %% creating the training label ground truth
training_instance_matrix = images(trIdx,:); %% creating the training data matrix
test_label_vector = a(teIdx); %% creating the testing label ground truth
test_instance_matrix = images(teIdx,:); %% creating the test data matrix

%---------------------------------------------------training SVM using a RBF kernel
% model = svmtrain(training_label_vector, training_instance_matrix, '-t 2 -g 0.0001');
% model = svmtrain(training_label_vector, training_instance_matrix, '-t 2 -g 0.001');
% model = svmtrain(training_label_vector, training_instance_matrix, '-t 2 -g 0.01');
% model = svmtrain(training_label_vector, training_instance_matrix, '-t 2 -g 0.04');
% model = svmtrain(training_label_vector, training_instance_matrix, '-t 2 -g 0.07');
% model = svmtrain(training_label_vector, training_instance_matrix, '-t 2 -g 0.1');
% model = svmtrain(training_label_vector, training_instance_matrix, '-t 2 -g 1');

% model = svmtrain(training_label_vector, training_instance_matrix, '-t 2 -g 0.02');
model = svmtrain(training_label_vector, training_instance_matrix, '-t 2 -g 0.03');
% model = svmtrain(training_label_vector, training_instance_matrix, '-t 2 -g 0.04');
% model = svmtrain(training_label_vector, training_instance_matrix, '-t 2 -g 0.05');
% model = svmtrain(training_label_vector, training_instance_matrix, '-t 2 -g 0.06');

%---------------------------------------------------training SVM using a linear kernel
% model = svmtrain(training_label_vector, training_instance_matrix, '-t 0');

% SVM predicting
[predict_label, accuracy, dec_values] = svmpredict(test_label_vector, test_instance_matrix, model);
accuracy_avg = accuracy_avg + accuracy;
predict_label_total = [predict_label_total; predict_label];
test_label_vector_total = [test_label_vector_total; test_label_vector];
end

accuracy_final = accuracy_avg/5;
disp(strcat('Average accuracy after 5-fold cross-validation: ',num2str(accuracy_final(1,1)),'%'));

%ROC curve
auc = ROC(predict_label_total,test_label_vector_total);
disp(strcat('AUC:',num2str(auc)));