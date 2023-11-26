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

% 定义前馈网络
network = feedforwardnet(30,'traingd');
% network = feedforwardnet([15,15],'traingd');

images2 = images';
%-----------------------------------5-fold cross validation setting
accuracy_avg = 0;
for i = 1:5
trIdx = cvo.training(i); %% get the index of training samples
teIdx = cvo.test(i); %% get the index of the test samples
training_label_vector = a(trIdx); %% creating the training label
%ground truth
training_instance_matrix = images2(trIdx,:); %% creating the training
%data matrix
test_label_vector = a(teIdx); %% creating the testing label
%ground truth
test_instance_matrix = images2(teIdx,:); %% creating the test data

%---------------------------------------------------------neural network
network.divideParam.trainRatio = 1; % training set [%]
network.divideParam.valRatio = 0; % validation set [%]
network.divideParam.testRatio = 0; % test set [%]
network.inputs{1}.processFcns = {}; % modify the process function for inputs
network.outputs{2}.processFcns = {}; % modify the process function for outputs
network.layers{1}.transferFcn = 'logsig'; % the transfer function for the first layer
network.layers{2}.transferFcn = 'logsig'; % the transfer function for the second layer
% network.layers{3}.transferFcn = 'logsig'; % the transfer function for the second layer
network.trainParam.lr = 0.1; % learning rate

network = train(network, training_instance_matrix', training_label_vector');

Y = (sim(network, test_instance_matrix')).';

Y(Y > 0.5) = 1;
Y(Y <= 0.5) = 0;

correct = 0;
for j = 1:120
    if Y(j,1) == test_label_vector(j,1)
        correct = correct + 1;
    end
end
accuracy = correct/120;
disp(accuracy);
accuracy_avg = accuracy_avg + accuracy;
end
accuracy_final = (accuracy_avg/5)*100;
disp(strcat('Average accuracy after 5-fold cross-validation: ',num2str(accuracy_final),'%'));

%ROC curve
auc= ROC(Y,test_label_vector);  
disp(strcat('AUC:',num2str(auc)));