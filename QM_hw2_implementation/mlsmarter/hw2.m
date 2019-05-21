addpath('./data');
addpath('./minFunc');
%% FULL FEATURES
clear all;
load('SmarterML_Training_1250.mat');
X = feature;
y = label+1;

%% SELECTED FEATURES
clear all;
load('SmarterML_Training_1250_SF.mat');
X = feature;
y = label+1;
%% ============================== Describe ==============================
plot(X(y==2,34), X(y==2,26), 'b+'); hold on;
plot(X(y==1,34), X(y==1,26), 'ro')
legend('investor', 'Non-investor')
xlabel('Feature 34'); ylabel('Feature 26')

%% ================================= PCA =================================
X = zscore(X);
selected_X = [X(:, 21) X(:, 6) X(:, 34) X(:, 26) X(:, 42)]
[coefs, Xpca, variances] = pca(selected_X);
figure; plotData(Xpca(:,1:2), y)
title('PCA'); legend('Investor', 'Non-investor')

%% ================================= LDA =================================
[Xlda, mapping] = mylda(X, y, 2);
figure; plotData(Xlda, y);
title('LDA'); legend('0', '1')

%% ================================ t-SNE ================================
[Xtsne, loss] = tsne(X);
figure; plotData(Xtsne, y)
title('t-SNE'); legend('Investor', 'Non-investor')

%% ============== Logistic Regression for multiple classes ===============
% [m n] = size(X);
% rand_indices = randperm(m);
% figure; displayData(X(rand_indices(1:100), :));

seed = 1;
[Xtrain, Xval, Xtest] = splitData(X, [0.8; 0.1; 0.1], seed);
[ytrain, yval, ytest] = splitData(y, [0.8; 0.1; 0.1], seed);

lambda = 0.005;
all_theta = trainLogisticReg(Xtrain, ytrain, lambda);

ypredtrain = predictLogisticReg(all_theta, Xtrain);
ypredval = predictLogisticReg(all_theta, Xval);
ypredtest = predictLogisticReg(all_theta, Xtest);
disp('Logistic Regression');
fprintf('Train Set Accuracy: %f\n', mean(ypredtrain==ytrain)*100) ;
fprintf('Validation Set Accuracy: %f\n', mean(ypredval==yval)*100);
fprintf('Test Set Accuracy: %f\n', mean(ypredtest==ytest)*100);

load('SmarterML_eval.mat');
X_eval = feature;
result_eval = predictLogisticReg(all_theta, X_eval) - 1;
fileName={'eval_Logistic_Regression.txt'};
fid=fopen('eval_Logistic_Regression.txt','w');
fprintf(fid, '%d\n' ,result_eval);
fclose(fid);
% figure; displayData(Xtest(ypredtest~=ytest, :));

%% ============= Softmax classification for multiple classes =============
X = X'; y = y';

seed = 1;
[Xtrain, Xval, Xtest] = splitData(X, [0.75 0.2 0.05], seed);
[ytrain, yval, ytest] = splitData(y, [0.75 0.2 0.05], seed);

numClasses = 2;
initial_theta = reshape(0.005 * randn(numClasses, size(X,1)), [], 1);

lambda = 1e-4;
[cost,grad] = costSoftmax(initial_theta, Xtrain(:,1:10), ytrain(1:10), numClasses, lambda);
numGrad = checkGradient( @(p) costSoftmax(p, Xtrain(:,1:10), ytrain(1:10), numClasses, lambda), initial_theta);
diff = norm(numGrad-grad)/norm(numGrad+grad) % Should be less than 1e-9

lambda = 0.01;
options = struct('display', 'on', 'Method', 'lbfgs', 'maxIter', 400);
theta = trainSoftmax(Xtrain, ytrain, numClasses, lambda, options);

ypredtrain = predictSoftmax(theta, Xtrain, numClasses);
ypredval = predictSoftmax(theta, Xval, numClasses);
ypredtest = predictSoftmax(theta, Xtest, numClasses);
fprintf('Train Set Accuracy: %f\n', mean(ypredtrain==ytrain)*100) ;
fprintf('Validation Set Accuracy: %f\n', mean(ypredval==yval)*100);
fprintf('Test Set Accuracy: %f\n', mean(ypredtest==ytest)*100);

%% ==================== Implementing Neural network ====================
clear  all;

% X = randn(8, 100);
% y = randi(10, 1, 100);
% 
% parameters = []; % Reset the variable parameters
% parameters.lambda = 0; % Weight decay penalty parameter
% parameters.beta = 1; % sparsity penalty parameter
% [theta thetaSize] = initNNParameters(8, 5, 10);
%  
% [cost,grad] = costNeuralNetwork(theta, thetaSize, X, y, parameters);
% numGrad = checkGradient( @(p) costNeuralNetwork(p, thetaSize, X, y, parameters), theta);
% diff = norm(numGrad-grad)/norm(numGrad+grad) % Should be less than 1e-9

load('SmarterML_Training_1250_SF.mat'); % Gives X, y
X = feature';
y = label+1
y=y'
disp(size(X));
disp(size(y));

%X = X'; y = y';
[Xtrain, Xval, Xtest] = splitData(X, [0.7 0.2 0.1], 0);
[ytrain, yval, ytest] = splitData(y, [0.7 0.2 0.1], 0);

parameters = []; % Reset the variable parameters
parameters.lambda = 1e-3; % This is a tunable hyperparameter
parameters.beta = 0.8; % This is a tunable hyperparameter
numhid = 6; % % This is a tunable hyperparameter

numvis = size(X, 1);
numout = length(unique(y));
[theta thetaSize] = initNNParameters(numvis, numhid, numout);

costFunction = @(p) costNeuralNetwork(p, thetaSize, Xtrain, ytrain, parameters);

tic
options = struct('display', 'on', 'Method', 'lbfgs', 'MaxIter', 1000); %400 number of iter
[optTheta, optCost] = fmincg(costFunction, theta, options);
toc

[W1 W2 b1 b2] = theta2params(optTheta, thetaSize);
% displayData(W1);

ypredtrain = predictNeuralNetwork(optTheta, thetaSize, Xtrain);
ypredval = predictNeuralNetwork(optTheta, thetaSize, Xval);
ypredtest = predictNeuralNetwork(optTheta, thetaSize, Xtest);
disp('Neural network - Selected Features')
fprintf('lambda: %f\n', parameters.lambda);
fprintf('beta: %f\n', parameters.beta);
fprintf('number of hidden layers: %d\n', numhid);
fprintf('Train Set Accuracy: %f\n', mean(ypredtrain==ytrain)*100);
fprintf('Val Set Accuracy: %f\n', mean(ypredval==yval)*100);
fprintf('Test Set Accuracy: %f\n', mean(ypredtest==ytest)*100);

load('SmarterML_eval_SF.mat');
X_eval = feature';
result_eval = predictNeuralNetwork(optTheta, thetaSize, X_eval) - 1;
fileName={'eval_Neural_Netrork_SF.txt'};
fid=fopen('eval_Neural_Network_SF.txt','w');
fprintf(fid, '%d\n' ,result_eval);
fclose(fid);

%% APPS - FULL FEATURES
clear  all;
load('SmarterML_Training_1000.mat');
X = feature';
y = label+1
y=y'
data =[X;y];

%% APPS - SELECTED FEATURES
clear  all;
load('SmarterML_Training_1000_SF.mat');
X = feature';
y = label+1
y=y'
data =[X;y];

%% TEST - FULL-FEATURES
load('SmarterML_testing.mat');
X_test = feature;
y_test = label+1;

acc_mean = mean(predict(trainedModel_Tree.ClassificationTree,X_test)==y_test);
result_test = predict(trainedModel_Tree.ClassificationTree,X_test)-1;
fileName={'Tree_test.txt'};
fid=fopen('Tree_test.txt','w');
fprintf(fid, '%d\n' ,result_test);
fclose(fid);

disp('Decision Tree - Test accuracy:');
disp(acc_mean);

acc_mean = mean(predict(trainedModel_Boosted_Trees.ClassificationEnsemble,X_test)==y_test);
result_test = predict(trainedModel_Boosted_Trees.ClassificationEnsemble,X_test)-1;
fileName={'Boosted_Trees_test.txt'};
fid=fopen('Boosted_Trees_test.txt','w');
fprintf(fid, '%d\n' ,result_test);
fclose(fid);

disp('Boosted Trees (Ensemble) - Test accuracy:');
disp(acc_mean);

acc_mean = mean(predict(trainedModel_Discriminant.ClassificationDiscriminant,X_test)==y_test);
result_test = predict(trainedModel_Discriminant.ClassificationDiscriminant,X_test)-1;
fileName={'Discriminant_test.txt'};
fid=fopen('Discriminant_test.txt','w');
fprintf(fid, '%d\n' ,result_test);
fclose(fid);

disp('Linear Discriminant - Test accuracy:');
disp(acc_mean);

acc_mean = mean(predict(trainedModel_KNN.ClassificationKNN,X_test)==y_test);
result_test = predict(trainedModel_KNN.ClassificationKNN,X_test)-1;
fileName={'KNN_test.txt'};
fid=fopen('KNN_test.txt','w');
fprintf(fid, '%d\n' ,result_test);
fclose(fid);

disp('Cosin KNN - Test accuracy:');
disp(acc_mean);

acc_mean = mean(predict(trainedModel_SVM.ClassificationSVM,X_test)==y_test);
result_test = predict(trainedModel_SVM.ClassificationSVM,X_test)-1;
fileName={'SVM_test.txt'};
fid=fopen('SVM_test.txt','w');
fprintf(fid, '%d\n' ,result_test);
fclose(fid);

disp('Quadratic SVM - Test accuracy:');
disp(acc_mean);
%% EVALUATION - FULL FEATURES
load('SmarterML_eval.mat');
X_eval = feature;

result_eval = predict(trainedModel_Tree.ClassificationTree,X_eval)-1;
fileName={'eval_Decision_Tree.txt'};
fid=fopen('eval_Decision_Tree.txt','w');
fprintf(fid, '%d\n' ,result_eval);
fclose(fid);

result_eval = predict(trainedModel_Boosted_Trees.ClassificationEnsemble,X_eval)-1;
fileName={'eval_Boosted_Trees.txt'};
fid=fopen('eval_ensemble_Boosted_Trees.txt','w');
fprintf(fid, '%d\n' ,result_eval);
fclose(fid);

result_eval = predict(trainedModel_Discriminant.ClassificationDiscriminant,X_eval)-1;
fileName={'eval_discriminant.txt'};
fid=fopen('eval_discriminant.txt','w');
fprintf(fid, '%d\n' ,result_eval);
fclose(fid);

result_eval = predict(trainedModel_KNN.ClassificationKNN,X_eval)-1;
fileName={'eval_KNN.txt'};
fid=fopen('eval_KNN.txt','w');
fprintf(fid, '%d\n' ,result_eval);
fclose(fid);

result_eval = predict(trainedModel_SVM.ClassificationSVM,X_eval)-1;
fileName={'eval_SVM.txt'};
fid=fopen('eval_SVM_Tree.txt','w');
fprintf(fid, '%d\n' ,result_eval);
fclose(fid);

%% TEST FOR SELECTED FEATURES BASED METHODS
load('SmarterML_testing_250_SF.mat');
X_test_SF = feature;
y_test_SF = label+1;

acc_mean = mean(predict(trainedModel_Tree_v2.ClassificationTree,X_test_SF)==y_test_SF);
result_test_SF = predict(trainedModel_Tree_v2.ClassificationTree,X_test_SF)-1;
fileName={'Tree_test_SF.txt'};
fid=fopen('Tree_test_SF.txt','w');
fprintf(fid, '%d\n' ,result_test_SF);
fclose(fid);

disp('Selected Features - Decision Tree - Test accuracy:');
disp(acc_mean);

acc_mean = mean(predict(trainedModel_Boosted_Trees_v2.ClassificationEnsemble,X_test_SF)==y_test_SF);
result_test_SF = predict(trainedModel_Boosted_Trees_v2.ClassificationEnsemble,X_test_SF)-1;
fileName={'Boosted_Trees_test_SF.txt'};
fid=fopen('Boosted_Trees_test_SF.txt','w');
fprintf(fid, '%d\n' ,result_test_SF);
fclose(fid);

disp('Selected Features - Boosted Trees (Ensemble) - Test accuracy:');
disp(acc_mean);

acc_mean = mean(predict(trainedModel_Discriminant_v2.ClassificationDiscriminant,X_test_SF)==y_test_SF);
result_test_SF = predict(trainedModel_Discriminant_v2.ClassificationDiscriminant,X_test_SF)-1;
fileName={'Discriminant_test_SF.txt'};
fid=fopen('Discriminant_test_SF.txt','w');
fprintf(fid, '%d\n' ,result_test_SF);
fclose(fid);

disp('Selected Features - Quadratic Discriminant - Test accuracy:');
disp(acc_mean);

acc_mean = mean(predict(trainedModel_KNN_v2.ClassificationKNN,X_test_SF)==y_test_SF);
result_test_SF = predict(trainedModel_KNN_v2.ClassificationKNN,X_test_SF)-1;
fileName={'KNN_test_SF.txt'};
fid=fopen('KNN_test_SF.txt','w');
fprintf(fid, '%d\n' ,result_test_SF);
fclose(fid);

disp('Selected Features - Weighted KNN - Test accuracy:');
disp(acc_mean);

acc_mean = mean(predict(trainedModel_SVM_v2.ClassificationSVM,X_test_SF)==y_test_SF);
result_test_SF = predict(trainedModel_SVM_v2.ClassificationSVM,X_test_SF)-1;
fileName={'SVM_test_SF.txt'};
fid=fopen('SVM_test_SF.txt','w');
fprintf(fid, '%d\n' ,result_test);
fclose(fid);

disp('Selected Features - Cubic SVM - Test accuracy:');
disp(acc_mean);

%% EVALUATION - SELECTED FEATURES
load('SmarterML_eval_SF.mat');
X_eval_SF = feature;

result_eval_SF = predict(trainedModel_Tree_v2.ClassificationTree,X_eval_SF)-1;
fileName={'eval_SF_Decision_Tree.txt'};
fid=fopen('eval_SF_Decision_Tree.txt','w');
fprintf(fid, '%d\n' ,result_eval_SF);
fclose(fid);

result_eval_SF = predict(trainedModel_Boosted_Trees_v2.ClassificationEnsemble,X_eval_SF)-1;
fileName={'eval_SF_Ensemble_Boosted_Trees.txt'};
fid=fopen('eval_SF_Ensemble_Boosted_Trees.txt','w');
fprintf(fid, '%d\n' ,result_eval_SF);
fclose(fid);

result_eval_SF = predict(trainedModel_Discriminant_v2.ClassificationDiscriminant,X_eval_SF)-1;
fileName={'eval_SF_discriminant.txt'};
fid=fopen('eval_SF_discriminant.txt','w');
fprintf(fid, '%d\n' ,result_eval_SF);
fclose(fid);

result_eval_SF = predict(trainedModel_KNN_v2.ClassificationKNN,X_eval_SF)-1;
fileName={'eval_SF_KNN.txt'};
fid=fopen('eval_SF_KNN.txt','w');
fprintf(fid, '%d\n' ,result_eval_SF);
fclose(fid);

result_eval_SF = predict(trainedModel_SVM_v2.ClassificationSVM,X_eval_SF)-1;
fileName={'eval_SF_SVM.txt'};
fid=fopen('eval_SF_SVM_Tree.txt','w');
fprintf(fid, '%d\n' ,result_eval_SF);
fclose(fid);
