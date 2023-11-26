clc;
clear;
close all;

% gamma = [0.0001,0.001,0.01,0.04,0.07,0.1,1];
% acc_compare = [68.8333,94,98.1667,98.5,98,85.8333,68.8333];
% auc_compare = [0.50418,0.91458,0.9795,0.97961,0.95985,0.78928,0.50392];

gamma = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07];
acc_compare = [98.1667,98.5,98.6667,98.5,97.6667,97.5,98];
auc_compare = [0.9795,0.98736,0.99145,0.98116,0.96993,0.97153,0.95985];

figure(2);
plot(gamma,acc_compare);
title('gamma-average accurancy');
xlabel('gamma');
ylabel('average accurancy(%)');


figure(3);
plot(gamma,auc_compare);
title('gamma-AUC');
xlabel('gamma');
ylabel('AUC');