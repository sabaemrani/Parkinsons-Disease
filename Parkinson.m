clear all
close all
% X is 
% number of tasks
NumTasks=12; 
% number of repetitions 
Nrep=20;
% splitting ratio of training set
training_percent = 0.7;

Weights=cell(Nrep,1);
NMSE=zeros(Nrep,NumTasks);
RMSE=zeros(Nrep,NumTasks);

X_trN=cell(1,NumTasks);
Y_trN=cell(1,NumTasks);
X_teN=cell(1,NumTasks);
Y_teN=cell(1,NumTasks);
for rep =1:Nrep
% splitting data randomly into traing and test sets
[X_tr, Y_tr, X_te, Y_te] = mtSplitPerc(X,Y,training_percent);
% Z-score normalizing the training and test set
for t=1:length(X)
    X_trN{t} = zscore(X_tr{t});
    Y_trN{t} = zscore(Y_tr{t});
    X_teN{t} = zscore(X_te{t});
    Y_teN{t} = zscore(Y_te{t});
end

% parameter are selected using cross validation
rho1=0.5;
rho2=1000;
rho3=10;

 for tt=1:NumTasks 
 X_trN{tt}=X_trN{tt}';
 end
% CFGLasso
W= Least_CFGLasso(X_trN, Y_trN, rho1, rho2, rho3);

for j=1:NumTasks
    Y_predn_j = X_teN{j}* W(:, j); 
    Y_pred_j = Y_predn_j*std(Y_tr{j})+mean(Y_tr{j});

RMSE_j=(sqrt( sum((Y_pred_j-Y_te{j}).^2 /length(Y_pred_j))))/(max(Y_te{j})-min(Y_te{j}));
NMSE_j = mean((Y_pred_j-Y_te{j}) .^ 2)/(mean(Y_te{j}) .* mean(Y_pred_j)); 
NMSE(rep,j)=NMSE_j;
RMSE(rep,j)=RMSE_j;

end

Weights{rep}=W;
end


NMSE_ave_tasks_CFGLasso_All= mean(NMSE);
NMSE_ave_CFGLasso_All=mean(NMSE_ave_tasks_CFGLasso_All);
NMSE_std_CFGLasso_All=std(NMSE_ave_tasks_CFGLasso_All);

RMSE_ave_tasks_CFGLasso_All= mean(RMSE);
RMSE_ave_CFGLasso_All=mean(RMSE_ave_tasks_CFGLasso_All);
RMSE_std_CFGLasso_All=std(RMSE_ave_tasks_CFGLasso_All);

% weights
WAll=zeros(length(X{1}(1,:)),NumTasks);
for rep=1:Nrep
WAll=WAll+Weights{rep};
end
W_ave=WAll./Nrep; % average weight over different repetitions of experiment
W4=abs(W_ave(2:end,:)); % the first column is the MDS_UPDRS at baseline 

for i=1:length(W4)

W_norm=(W4(:,i)-min(W4(:,i)))/(max(W4(:,i))-min(W4(:,i)));
figure; bar(W_norm(2:end))
end




