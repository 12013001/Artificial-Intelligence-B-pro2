function  auc = ROC(predict, ground_truth)
x = 1.0;
y = 1.0;
%计算positive和negative的样本数量
num_positive = sum(ground_truth==1);
num_negative = sum(ground_truth==0);
%计算图像步长
ystep = 1.0/num_positive;
xstep = 1.0/num_negative;
%排序输出结果
[n,index] = sort(predict);
ground_truth = ground_truth(index);
%画出ROC曲线
for i=1:length(ground_truth)
    if ground_truth(i) == 1
        y = y - ystep;
    else
        x = x - xstep;
    end
    X(i)=x;
    Y(i)=y;
end
hold on;
xlim([0 1]);
ylim([0 1]);
plot(X,Y);
plot(X,X,'--k');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve');
grid on;
%计算ROC曲线下面积，AUC
auc = -trapz(X,Y);
legend(strcat('ROC curve (area = ',num2str(auc),' )'));
end