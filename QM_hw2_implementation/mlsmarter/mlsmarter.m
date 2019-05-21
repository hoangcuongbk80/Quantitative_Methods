% ================================================================
% Introduction to Machine learning
% SMARTER programme
% ?rebro University
% 
% MATLAB Tips:
% You can read about any MATLAB function by using the help function, e.g.:
% >> help plot
% To run a single line of code simply highlight the code you want to run 
% and press F9. 

%% Preparation
cd('M:\Quantitative-Methods\mlsmarter') %uncomment and change 
addpath('.\data');
addpath('.\minFunc');

%% ============== Part 1: Linear Regression with one variable =============
load carbig.mat
% This command loads a number of variables to the MATLAB workspace. The 
% variables contain information about cars such as WEIGHT, Horsepower, 
% Model_Year e.t.c. The task is to fit a linear model that predicts the 
% horsepower based on the weight of the car.

x = Weight; % data vector
y = Horsepower; % prediction values

% The first thing to do when working with a new data set is to plot it. For 
% this data set you don't want to draw any lines between each data point so 
% set the plot symbol to point "." Type ">> help plot" to see how.
plot(x,y,'.','Color','b')
xlabel('Weight [kg]');
ylabel('Horsepower [hp]');

% Next we need to clean up the data and remove any training data that
% contains any NaN (not-a-number) or Inf (infinity) values. 
[x y] = RemoveData(x, y);

% Scale the features down using mean normalization. 
[x mu sigma] = featureMeanNormalize(x);

% ======= Linear Regression: Cost function and derivative ========
X = [ones(size(x,1), 1), x]; % Add a column of ones to the design matrix X
initial_theta = zeros(2, 1); % initialize fitting parameters

% Compute cost and derivative for a given theta. Complete the code in 
% costLinearRegression.m before continuing.
[J grad] = costLinearRegression(initial_theta, X, y);

% You can check if your solution is correct by calculating the numerical
% gradients (numgrad) and compare them with the analytical gradients 
% (grad). The code in checkGradient.m is already completed so you don't 
% need to change anything there.
numgrad = checkGradient(@(p) costLinearRegression(p, X, y), initial_theta);

% If your implementation is correct the two columns should be very similar. 
disp([numgrad grad]); 
% and the difference between the gradients should be less than 1e-9
fprintf('Difference between analytical and numerical gradients: %.3e\n',...
    norm(numgrad-grad)/norm(numgrad+grad))

% Set hyperparameters for gradient descent 
num_iters = 500;
alpha = 0.01;

% Run Gradient Descent. 
[theta J_history] = gradientDescent(initial_theta, X, y, alpha, num_iters);

% You can check if your implementation is correct by comparing with the
% pre-build MATLAB function fminunc. This function automatically sets the
% learning rate alpha.
[theta_fminunc, cost] = fminunc(@(t)(costLinearRegression(t, X, y)), ...
    initial_theta, optimset('GradObj', 'on', 'MaxIter', 500));
fprintf('Difference between gradientDescent.m and fminunc: %e\n', ...
    norm(theta-theta_fminunc)/norm(theta+theta_fminunc))

% Plot J_history. If your implementation is correct J should decrease after
% each iteration.
figure;
plot(J_history);
xlabel('Iteration')
ylabel('Cost J(\theta)')

% Plot the data and the linear regression model
figure;
plot(X(:,2), y, 'b.'); hold on; 
plot([min(X(:,2)); max(X(:,2))],[1 min(X(:,2));1 max(X(:,2))]*theta, 'r-')
legend('Training data', 'Linear regression')

% Now let's try different values of alpha (the learning rate) and plot the history of J.
% Calculate the history of J for each of the following choices of alpha: 
% [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]. 
num_iters = 500;
alpha_values = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3];
colors = {'r' 'g' 'b' 'k' 'c' 'm'};
figure; hold on;
for i = 1:length(alpha_values)
    [theta J_history] = gradientDescent(initial_theta, X, y, alpha_values(i), num_iters);
    plot(1:num_iters, J_history, colors{i})
end
legend(num2str(alpha_values'))

%% ======= Part 2: Linear regression with multiple variables ==============
clear all; %Clear all variables in workspace

% Load data
load carbig.mat
x = [Weight MPG]; % We use two variable - weight and miles per gallon (MPG)
y = Horsepower; % prediction values

% Plot the data. Use Tools -> Rotate 3D to examine the data.
figure;
plot3(x(:,1),x(:,2),y,'.','Color','b')
xlabel('Weight [kg]');
ylabel('Fuel efficiency [MPG]');
zlabel('Horsepower [hp]');
grid on

% Remove pairs of data that contains any NaN values. Change the code in
% RemoveData.m so that it works for multiple variables.
[x y] = RemoveData(x, y);

% Normalize both feature vectors
[x mu sigma] = featureMeanNormalize(x);

% Initialize parameters
X = [ones(size(x,1), 1) x]; % Add intercept term to X
initial_theta = zeros(3, 1);

% Check gradients to see if implementation is correct. 
[J grad] = costLinearRegression(initial_theta, X, y);
numgrad = checkGradient(@(p) costLinearRegression(p, X, y), initial_theta);
diff = norm(numgrad-grad)/norm(numgrad+grad) % Should be less than 1e-9

% Hyperparameters for gradient descent 
alpha = 0.1;
num_iters = 1500;

% Run Gradient Descent
[theta J_history] = gradientDescent(initial_theta, X, y, alpha, num_iters);

% Plot J_history. If your implementation is correct J should decrease after
% each iteration.
figure;
plot(J_history)

% Plot the data and the linear regression model
figure;
plot3(X(:,2), X(:,3), y, 'b.'); hold on; 
range=-2:.1:2;
[xind,yind] = meshgrid(range);
Z = zeros(size(xind));
for i=1:size(xind,1) %rows
    for j=1:size(xind,2) %cols
        Z(i,j) = [1 xind(i,j) yind(i,j)]*theta;
    end
end
surf(xind,yind,Z)
shading flat; grid on;
xlabel('Normalized Weight'); 
ylabel('Normalized MPG'); 
zlabel('Horsepower [hp]')

% Predict how much horsepower a car would have that weights 3000 kg and has
% a MPG of 30. Here we need to normalize the data.
y_pred = [1 (3000-mu(1))/sigma(1) (30-mu(2))/sigma(2)]*theta

%% ================ Part 3: Normal Equation ================
% We can also find the linear regression model with the normal equation.
% Here it is not as important to normalize the data.
clear all; %Clear all variables in workspace

% Load data
load carbig.mat
x = [Weight MPG]; % We use two variable - weight and miles per gallon (MPG)
y = Horsepower; % prediction values
[x y] = RemoveData(x, y); % Remove bad training data
X = [ones(size(x,1), 1) x]; % Add intercept term to X

% Calculate the parameters from the normal equation. You need to finish the
% code in normalEqn.m first.
theta = normalEqn(X, y);

% Predict how much horsepower a car would have that weights 3000 kg and has 
% a MPG of 30. You should get the same answer as in Part 2. Here we don't
% need to normalize the data. 
y_pred_normalEqn = [1 3000 30]*theta


%% ==================== Part 4: Logistic Regression ====================
clear; % Clear all workspace variables
load hospital.mat
% This data set contain the information about age and blood
% pressure for 100 patients. Your task is to train a logistic regression
% classifier to classify whether a patient is a smoker or a non-smoker.

% Plot the data
plot(x(y==1,1), x(y==1,2), 'b+'); hold on;
plot(x(y==0,1), x(y==0,2), 'ro')
legend('Smoker', 'Non-smoker')
xlabel('Age'); ylabel('Blood Pressure')

% Implement the sigmoid function. Complete the code in sigmoid.m. The 
% function should perform the sigmoid function of each element in a vector
% or matrix. 
g = sigmoid([-10 0.3; 0 10])
% If your implementation is correct you should get the following answer.
% g =
% 
%     0.0000    0.5744
%     0.5000    1.0000

% Normalize both feature vectors
[x mu sigma] = featureMeanNormalize(x);

% Add intercept term to x and initialize theta
[m, n] = size(x);
X = [ones(size(x,1), 1) x];
initial_theta = zeros(n + 1, 1);

% Now it is time to implement logistic regression in
% costLogisticRegression.m. 
[J grad] = costLogisticRegression(initial_theta, X, y);

% You can check if your implementation is correct by comparing the
% gradients with the analytical gradients.
numgrad = checkGradient(@(p) costLogisticRegression(p, X, y), ...
    initial_theta);
diff = norm(numgrad-grad)/norm(numgrad+grad) % Should be less than 1e-9

% Instead of using our own implementation of gradient descent
% (gradentDescent.m) we will use the pre-built MATLAB function fminunc
% which sets the learning rate alpha automatically.
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = fminunc(@(t)(costLogisticRegression(t, X, y)), ...
    initial_theta, options);

% Plot data and decision boundary
figure;
plot(X(y==1,2), X(y==1,3), 'b+'); hold on
plot(X(y==0,2), X(y==0,3), 'ro');
plot_x = [min(X(:,2))-2,  max(X(:,2))+2];
plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));
plot(plot_x, plot_y, 'Color', 'k', 'Linewidth', 2)
xlabel('Age (normalized)'); ylabel('Blood Pressure (normalized)')
legend('Smoker', 'Non-smoker')

% Now use the learned logistic regression model to predict the probability 
% that a patient with age 32 and blood pressure 124 is a smoker. 
%prob = ...
prob = sigmoid([1 (32-mu(1))/sigma(1) (124-mu(2))/sigma(2)] * theta)


%% ============== Part 5: Polynomial features ==================
% In this part we introduce polynomial features in order to fit a curve
% instead of a line to the data. 
clear; % Clear all workspace variables
load hospital.mat

degree = 2;
Xpoly = mapFeature(x(:,1), x(:,2), degree);
% Feature normalization becomes important when using polynomic features
[Xpoly(:,2:end) mu sigma] = featureMeanNormalize(Xpoly(:,2:end)); 
initial_theta = zeros(size(Xpoly, 2), 1);

% Set options and optimize theta
options = optimset('GradObj', 'on', 'MaxIter', 100);
[theta, J, exit_flag, output] = fminunc(@(t)(costLogisticRegression(t, ...
    Xpoly, y)), initial_theta, options);

% Plot data and Boundary
plot(x(y==1,1), x(y==1,2), 'b+'); hold on
plot(x(y==0,1), x(y==0,2), 'ro');
u = linspace(15, 60, 50);
v = linspace(95, 150, 50);
z = zeros(length(u), length(v));
for i = 1:length(u)
    for j = 1:length(v)
        temp = (mapFeature(u(i), v(j), degree) - [1 mu])./[1 sigma];
        z(i,j) = sigmoid(temp*theta);
    end
end
z = z'; % important to transpose z before calling contour
contour(u, v, z, [0.5 0.5], 'LineWidth', 2)

% Now use the learned logistic regression model with polynomial features
% to predict the probability that a patient with age 32 and blood pressure 
% 124 is a smoker. Use mapFeature.m and mu and sigma to normalize the 
% values first.
%prob = ...
prob = sigmoid((mapFeature(32, 124, degree)- [1 mu])./[1 sigma] * theta)
% ============================================================



%% ============ Part 6: Dimensionality Reduction (PCA, LDA, t-sne) =========
clear all;
% PCA and LDA are very useful tools for plotting data that has more than 2
% dimensions. They can also be used for feature selection. 
% We will use the smallMNIST dataset to demonstrate how they work.
% This data set is used to classify handwritten digits 0-9. The
% file smallMNIST contains a smaller version of the popular MNIST data set.
% The variable X contains 5000 images (500 per digit). Each image is 20x20
% pixels big, so the total number of features per image is 400. The
% variable y is the labelvector where y(i) = k means instance i belongs to
% digit k except for k = 10 which represents the digit 0. 
load('smallMNIST.mat'); % Loads X and y

% Let's reduce the number of images to visualize the data better.
X = X(1:10:end,:);
y = y(1:10:end);

% Use the function imagesc to plot the first image in this data set X(1,:).
% You can use the function reshape to reshape this vector of 400 elements
% to a 20 x 20 matrix.

figure; imagesc(reshape(X(1,:), 20, 20)); colormap('gray')

% First we can normalize the data with mean normalization. We can use the
% matlab function zscore
X = zscore(X);

% Now we use the matlab pca function. Notice how PCA is an unsupervised
% algorithm and does not use the label vector y. 
[coefs, Xpca, variances] = pca(X);
% Plot the projected data in two dimensions. We only use the first two 
% columns in Xpca
figure; plotData(Xpca(:,1:2), y)
title('PCA'); legend('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

% Next we use LDA on the same data. We will use an implementation of LDA 
% from the Matlab Toolbox for Dimensionality Reduction. Notice how lda also 
% takes the label vector y as input.
[Xlda, mapping] = mylda(X, y, 2);

% Plot the projected data from LDA.
figure; plotData(Xlda, y);
title('LDA'); legend('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

% Now we do the same for tsne
[Xtsne, loss] = tsne(X);
figure; plotData(Xtsne, y)
title('t-SNE'); legend('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')


%% ============= Part 7: Gaussian Mixture Model (GMM) =====================
% We will use MATLABs GMM implementation for this part. 
clear all

load simplecluster_dataset
[~, y] = max(simpleclusterTargets',[],2); 
X = simpleclusterInputs';
plotData(X,y)

% Fit k number of gaussian mixture models to the data X. Plot a contour
% plot of the found mixure models. Type doc gmdistribution.fit to learn
% about this function. Run any example code there and change the necessary
% parameters to fit our data X (e.g., increase the maximum iterations,
% change the number of components, calculate the contour from [-1 2] on
% both x and y axis).
options = statset('Display','final', 'MaxIter', 1000);
obj = gmdistribution.fit(X,4,'Options',options);
hold on; h = ezcontour(@(x,y)pdf(obj,[x y]),[-1 2],[-1 2]); 

% Next we plot which cluster each data point in a grid would be assigned to
[Xgrid Ygrid] = meshgrid(-1:0.1:2, -1:0.1:2);
[idx,~ , p] = cluster(obj,[Xgrid(:), Ygrid(:)]);
plotData([Xgrid(:), Ygrid(:)], idx)

%% =============== Part 8: Hidden Markov Model (HMM) ======================
% Now we train a HMM to do temporal smoothing on predicted sleep stages.
clear

load sleepdata % Gives xtrain, ytrain, xtest, ytest

% First we plot the data. Each sample is a segment of 5 seconds. The y
% values are the predicted sleep stage where 
% 5 = awake stage
% 4 = REM sleep
% 3 = stage 1 sleep
% 2 = stage 2 sleep
% 1 = deep sleep
% ytrain is the correct sleep stages labeled by a human expert. xtrain is a
% predicted sleep stage from a machine learning algorithm. 
figure; hold on;
plot(ytrain, 'Color', 'r')
plot(xtrain, 'Color', 'b')

% It can be seen that the predicted sleep stage (blue line) changes a lot.
% We know from ytrain that such a sleep pattern is not very probable. 

% We first train a HMM on this data and then used the trained HMM to smooth
% out the predictions from another patient. The predicted stage will be our
% observations and the hidden state is the true stage. 
[TransitionMatrix, EmissionMatrix] = hmmestimate(xtrain, ytrain, ...
    'PSEUDOTRANSITIONS', ...
    0.001*ones(5, 5), 'PSEUDOEMISSIONS',0.001*ones(5, 5));

% Next we use the learned transition matrix and emission matrix to smooth
% out the predictions on the test patient.
ytest_pred = hmmviterbi(xtest,TransitionMatrix, EmissionMatrix)';

% Now we plot the true sleep stage and the predicted smoothed out sleep 
% stage. Now we see that the predicted stage is not changing as much.
figure; hold on;
plot(ytest, 'Color', 'r')
plot(ytest_pred, 'Color', 'b')
legend('True label', 'Predicted label')


%% =============== Part 9a: Support vector machines (SVM) =================
% We will use MATLABs version of the SVM. In reality, there are better
% packages out there that better handles large data sets. One of them is
% LIBSVM (http://www.csie.ntu.edu.tw/~cjlin/libsvm/). On their website
% there is a nice guide that you can read (A practical guide to SVM
% classification). But for now MATLABs version will do.

clear all;

% First we show how a linear SVM can be trained using on the hospital data
% from the previous lab.
load hospital.mat

% Train a linear SVM. With 2-dimensional data the svmtrain function can
% plot the data and learned model for us using the 'showplot' argument.
C = 1;
%kernel_fcn = 'linear';
kernel_fcn = 'rbf';
svmmodel = fitcsvm(x,y, 'KernelFunction', kernel_fcn, 'boxconstraint', C);

% Calculate accuracy on training data. Try change the KernelFunction and
% the hyperparameter C
y_pred = predict(svmmodel, x);

fprintf('Accuracy on training set: %0.2f\n', mean(y(:)==y_pred(:)))


%%  =============== Part 9b: Support vector machines (SVM) =================

%  Now we will use the SVM to classify emails into Spam or Non-Spam.
clear all;

% First we need to convert the email to a feature vector. You need to
% complete the code in processEmail.m and emailFeatures.m before 
% continuing. 
file_contents = readFile('emailSample1.txt');
word_indices  = processEmail(file_contents);
features = emailFeatures(word_indices)';

% The length of the feature vector should be 1899 and the number of
% non-zero features for this email example should be 45.
fprintf('Length of feature vector: %d\n', length(features)); 
fprintf('Number of non-zero entries: %d\n', sum(features > 0));

% The file spamTrain.mat contains the feature vector for 4000 emails stored
% in a matrix X and the labels y. The file spamTest.mat contains 1000
% emails that we will use as test set. 
load('spamTrain.mat'); % Gives X, y to your workspace
load('spamTest.mat'); % % Gives Xtest, ytest to your workspace

% Train a linear SVM on the training set. You might need to change the
% number of iterations. Use svmclassify to calculate the classification
% accuracy on the testing set and the training set. Use help svmclassify as
% usual when encountering a new matlab function. 
C = 0.1;
svmmodel = fitcsvm(X,y, 'KernelFunction', 'linear', 'boxconstraint', C);
p_train = predict(svmmodel, X);
p_test = predict(svmmodel, Xtest);

% Now calculate the accuracy on the trainin and testing set. You should get 
% close to 100% on the training set and a bit lower on the testing set.
% Change the paramter C to get as high accuracy as possible on the testing
% set (at least 97%).
fprintf('Training Accuracy: %f\n', mean(double(p_train == y)) * 100);
fprintf('Test Accuracy: %f\n', mean(double(p_test == ytest)) * 100);

%  Since the model we are training is a linear SVM, we can inspect the
%  weights learned by the model to understand better how it is determining
%  whether an email is spam or not. The following code finds the words with
%  the highest weights in the classifier. Informally, the classifier
%  'thinks' that these words are the most likely indicators of spam.
[weight, idx] = sort(svmmodel.Alpha, 'descend');
vocabList = getVocabList();

fprintf('\nTop predictors of spam: \n');
for i = 1:15
    fprintf(' %-15s (%f) \n', vocabList{idx(i)}, weight(i));
end

% Now we try the trained spam classifier on our own email. The file 
% emailSample2.txt contains one of my academic spam emails. Let's see if 
% our classifier can correctly classify it.
filename = 'emailSample2.txt';
file_contents = readFile(filename);
word_indices  = processEmail(file_contents);
x = emailFeatures(word_indices)';
p = predict(svmmodel, x);
fprintf('\nProcessed %s\n\nSpam Classification: %d\n', filename, p);
fprintf('(1 indicates spam, 0 indicates not spam)\n\n');


%% ========= Part 10: Regularization for Logistic Regression =============
% First we will add regularization to our previously written logistic
% regression. Copy costLogisticRegression.m, sigmoid.m, and checkGradient.m
% from lab 1 and implement L2 weight decay regularization in 
% costLogisticRegression.m. 
clear all;

% We will test if regularization can help an email spam detector. 
load spamTrain
X = [ones(size(X,1), 1) X]; % add intercept term

% The test set is located in spamTest.mat. We need to divide the
% training data into train and validation sets. 
[Xtrain, Xval] = splitData(X, [0.8; 0.2], 0);
[ytrain, yval] = splitData(y, [0.8; 0.2], 0);
initial_theta = zeros(size(X, 2), 1);
clear X

% Implement L2 weight decay regularization in costLogisticRegression.m. 
% Do not regularize the first element in theta. Check the gradients on a
% small subset of the data.
lambda = 1e-4;
[J grad] = costLogisticRegression(initial_theta, Xtrain(1:10,:), yval(1:10), lambda);
numgrad = checkGradient(@(p) costLogisticRegression(p, Xtrain(1:10,:), yval(1:10), lambda), initial_theta);
diff = norm(numgrad-grad)/norm(numgrad+grad) % Should be less than 1e-9 (I got 2.0443e-12 with lambda = 1e-4)

% Train a logistic regression model on the full train set. We have
% provided two choices for optimization solver; minFunc and fmincg. You can 
% also use Matlab's fminunc or write your own similar to gradientDescent.m 
% from lab 1.
options = struct('display', 'on', 'Method', 'lbfgs', 'maxIter', 400);
lambda = 1e-4;
%q = fmincg(@(p)(costLogisticRegression(p, Xtrain, ytrain, lambda)), initial_theta, options);
theta = minFunc(@(p)(costLogisticRegression(p, Xtrain, ytrain, lambda)), initial_theta, options);

% Calculate the classification accuracy on the train and validation set
trainaccuracy = mean(round(sigmoid(Xtrain*theta))==ytrain)
valaccuracy = mean(round(sigmoid(Xval*theta))==yval)

% The classification error is calculated as the 1-accuracy.
trainerror = 1 - trainaccuracy
valerror = 1 - valaccuracy

% Make a plot of the classification error on the train and validation sets
% as a function of lambda. Try the following values for lambda:
% lambda_list = [0 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1]. You can use
% set(gca,'Xtick', 1:length(lambda_list), 'Xticklabel', lambda_list)
% to set the x-label as the values for lambda

%minFuncOptions = struct('display', 'on', 'Method', 'lbfgs', 'maxIter', 400);
fmincgOptions = struct('MaxIter', 400);
lambda_list = [0 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1];
%lambda_list = [0:0.01:0.1];
%minFuncresultstrain = zeros(size(lambda_list));
%minFuncresultstest = zeros(size(lambda_list));
fmincgresultstrain = zeros(size(lambda_list));
fmincgresultstest = zeros(size(lambda_list));
for i=1:length(lambda_list)
    lambda = lambda_list(i);
    %minFunctheta = minFunc(@(p) costLogisticRegression(p, Xtrain, ytrain, lambda), initial_theta, minFuncOptions);
    fmincgtheta = fmincg (@(p)(costLogisticRegression(p, Xtrain, ytrain, lambda)), initial_theta, fmincgOptions);
    %minFuncresultstrain(i) = 1-mean(round(sigmoid(Xtrain*minFunctheta))==ytrain);
    %minFuncresultstest(i) = 1-mean(round(sigmoid(Xval*minFunctheta))==yval);
    fmincgresultstrain(i) = 1-mean(round(sigmoid(Xtrain*fmincgtheta))==ytrain);
    fmincgresultstest(i) = 1-mean(round(sigmoid(Xval*fmincgtheta))==yval);
end
figure; hold on;
%plot(minFuncresultstest, 'Color', 'r')
%plot(minFuncresultstrain, 'Color', 'k')
plot(fmincgresultstest, 'r--')
plot(fmincgresultstrain, 'k--')
set(gca,'Xtick', 1:length(lambda_list), 'Xticklabel', lambda_list)
%legend('Jcv minFunc', 'Jtrain minFunc', 'Jcv fmincg', 'Jtrain fmincg')
legend('Jcv fmincg', 'Jtrain fmincg')
ylabel('error')
xlabel('amount of regularization')

% Calculate the classification accuracy on the test set
load spamTest
Xtest = [ones(size(Xtest,1),1) Xtest];
lambda = 0.0001;
lambda = 0;
fmincgtheta = fmincg (@(p)(costLogisticRegression(p, Xtrain, ytrain, lambda)), initial_theta, fmincgOptions);
mean(round(sigmoid(Xtest*fmincgtheta))==ytest)



%% ========= Part 11: Logistic Regression for multiple classes =============
% In this part we will train a logistic regression classifier for the task
% of classifying handwritten digits [0-9]. 
clear all;

% First we load the data from the file smallMNIST.mat which is a reduced 
% set of the MNIST handwritten digit dataset. The full data set can be
% downloaded from http://yann.lecun.com/exdb/mnist/. Our data X consist of 
% 5000 examples of 20x20 images of digits between 0 and 9. The number "0" 
% has the label 10 in the label vector y. The data is already normalized.
load('smallMNIST.mat'); % Gives X, y

% We use displayData to view 100 random examples at once. 
[m n] = size(X);
rand_indices = randperm(m);
figure; displayData(X(rand_indices(1:100), :));

% Now we divide the data X and label vector y into training, validation and
% test set. We use the same seed so that we dont get different
% randomizations. We will use hold-out cross validation to select the
% hyperparameter lambda.
seed = 1;
[Xtrain, Xval, Xtest] = splitData(X, [0.6; 0.3; 0.1], seed);
[ytrain, yval, ytest] = splitData(y, [0.6; 0.3; 0.1], seed);

% Now we train 10 different one vs all logistic regressors. Complete the
% code in trainLogisticReg.m before continuing. 
lambda = 0.01;
all_theta = trainLogisticReg(Xtrain, ytrain, lambda);

% Now we calculate the predictions using all 10 models. Finish the code in
% predictLogisitcReg.m before continuing. 
ypredtrain = predictLogisticReg(all_theta, Xtrain);
ypredval = predictLogisticReg(all_theta, Xval);
ypredtest = predictLogisticReg(all_theta, Xtest);
fprintf('Train Set Accuracy: %f\n', mean(ypredtrain==ytrain)*100) ;
fprintf('Validation Set Accuracy: %f\n', mean(ypredval==yval)*100);
fprintf('Test Set Accuracy: %f\n', mean(ypredtest==ytest)*100);

% It could be interesting to plot the missclassified examples.
figure; displayData(Xtest(ypredtest~=ytest, :));


%% ========= Part 12: Softmax classification for multiple classes ==========
% In this part we will train a softmax classifier for the task of 
% classifying handwritten digits [0-9]. 
clear  all;
addpath('./minFunc');

% Load the same data set. In softmax and neural networks the convention is 
% to let each column be one training input instead of each row as we have 
% previously used. 
load('smallMNIST.mat'); % Gives X, y
X = X'; y = y';

% Split into train, val, and test sets
seed = 1;
[Xtrain, Xval, Xtest] = splitData(X, [0.6 0.3 0.1], seed);
[ytrain, yval, ytest] = splitData(y, [0.6 0.3 0.1], seed);

% Initialize theta
numClasses = 10; % Number of classes
initial_theta = reshape(0.005 * randn(numClasses, size(X,1)), [], 1);

% For debugging purposes create a small randomized data matrix and
% labelvector. Calculate cost and grad and check gradients. Finish the code 
% in costSoftmax.m first. If your gradients don't match at first, try setting 
% lambda = 0; to see if the problem is with the error term or the 
% regularization term.
lambda = 1e-4;
[cost,grad] = costSoftmax(initial_theta, Xtrain(:,1:10), ytrain(1:10), numClasses, lambda);
numGrad = checkGradient( @(p) costSoftmax(p, Xtrain(:,1:10), ytrain(1:10), numClasses, lambda), initial_theta);
diff = norm(numGrad-grad)/norm(numGrad+grad) % Should be less than 1e-9

% Now we train the softmax classifier. Complete the code in trainSoftmax.m 
% before continuing. 
lambda = 0.01;
options = struct('display', 'on', 'Method', 'lbfgs', 'maxIter', 400);
theta = trainSoftmax(Xtrain, ytrain, numClasses, lambda, options);

% Now we calculate the predictions. Finish the code in
% predictSoftmax.m before continuing. 
ypredtrain = predictSoftmax(theta, Xtrain, numClasses);
ypredval = predictSoftmax(theta, Xval, numClasses);
ypredtest = predictSoftmax(theta, Xtest, numClasses);
fprintf('Train Set Accuracy: %f\n', mean(ypredtrain==ytrain)*100) ;
fprintf('Validation Set Accuracy: %f\n', mean(ypredval==yval)*100);
fprintf('Test Set Accuracy: %f\n', mean(ypredtest==ytest)*100);


%% ============== Part 13: Implementing Neural network ====================
% Time to implement a neural network. 
clear  all;

% Create a small randomized data matrix and labelvector for testing your 
% implementation.
X = randn(8, 100);
y = randi(10, 1, 100);

% Set Learning parameters. We start with coding the NN without any
% regularization
parameters = []; % Reset the variable parameters
parameters.lambda = 0; % Weight decay penalty parameter
parameters.beta = 1; % sparsity penalty parameter

% We initiliaze the network parameters assuming a small network of 8 input
% units, 5 hidden units, and 10 output units. 
[theta thetaSize] = initNNParameters(8, 5, 10);

% Calculate cost and grad and check gradients. 
[cost,grad] = costNeuralNetwork(theta, thetaSize, X, y, parameters);
numGrad = checkGradient( @(p) costNeuralNetwork(p, thetaSize, X, y, parameters), theta);
diff = norm(numGrad-grad)/norm(numGrad+grad) % Should be less than 1e-9


% Load the data set and split into train, val, and test sets.
load('smallMNIST.mat'); % Gives X, y
X = X'; y = y';
[Xtrain, Xval, Xtest] = splitData(X, [0.6 0.3 0.1], 0);
[ytrain, yval, ytest] = splitData(y, [0.6 0.3 0.1], 0);

% Set Learning parameters. 
parameters = []; % Reset the variable parameters
parameters.lambda = 1e-4; % This is a tunable hyperparameter
parameters.beta = 0.3; % This is a tunable hyperparameter
numhid = 50; % % This is a tunable hyperparameter

% Initiliaze the network parameters.
numvis = size(X, 1);
numout = length(unique(y));
[theta thetaSize] = initNNParameters(numvis, numhid, numout);

% Create "short hand" for the cost function to be minimized
costFunction = @(p) costNeuralNetwork(p, thetaSize, Xtrain, ytrain, parameters);

% Now, costFunction is a function that takes in only one argument (the 
% neural network parameters). Use tic and toc to see how long the training
% takes.
tic
options = struct('display', 'on', 'Method', 'lbfgs', 'MaxIter', 400);
[optTheta, optCost] = fmincg(costFunction, theta, options);
toc

% You can visualize what the network has learned by plotting the weights of
% W1 using displayData.
[W1 W2 b1 b2] = theta2params(optTheta, thetaSize);
displayData(W1);

% Now we predict all three sets. Complete the code in
% predictNeuralNetworkm. before continuing.
ypredtrain = predictNeuralNetwork(optTheta, thetaSize, Xtrain);
ypredval = predictNeuralNetwork(optTheta, thetaSize, Xval);
ypredtest = predictNeuralNetwork(optTheta, thetaSize, Xtest);

fprintf('Train Set Accuracy: %f\n', mean(ypredtrain==ytrain)*100);
fprintf('Val Set Accuracy: %f\n', mean(ypredval==yval)*100);
fprintf('Test Set Accuracy: %f\n', mean(ypredtest==ytest)*100);

%% ============== Part 14: Implementing Auto-encoder =======================
% Time to implement an auto-encoder. 

clear  all;

% Create a small randomized data matrix and labelvector
X = randn(8, 100);
y = randi(10, 1, 100);

% Set Learning parameters. 
parameters = []; % Reset the variable parameters
parameters.lambda = 1; % Weight decay penalty parameter
parameters.beta = 1; % sparsity penalty parameter

% We initiliaze the network parameters assuming a small network of 8 input
% units, 5 hidden units, and 8 output units (same as the number of input
% units).
[theta thetaSize] = initAEParameters(8, 5);

% Calculate cost and grad and check gradients. Note how costAutoencoder.m 
% does not require the label vector y.
[cost,grad] = costAutoencoder(theta, thetaSize, X, parameters);
numGrad = checkGradient( @(p) costAutoencoder(p, thetaSize, X, parameters), theta);
diff = norm(numGrad-grad)/norm(numGrad+grad) % Should be less than 1e-9


clear all;

load('smallMNIST.mat'); % Gives X, y
X = X'; y = y';
[Xtrain, Xval, Xtest] = splitData(X, [0.6 0.3 0.1], 0);
[ytrain, yval, ytest] = splitData(y, [0.6 0.3 0.1], 0);

% Set Learning parameters. 
parameters = []; % Reset the variable parameters
parameters.lambda = 1e-5; % This is a tunable hyperparameter
parameters.beta = 3; % This is a tunable hyperparameter
numhid = 50; % % This is a tunable hyperparameter

% Initiliaze the network parameters. Here we use initAEParameters.m
% instead.
numvis = size(X, 1);
[theta thetaSize] = initAEParameters(numvis, numhid);

% Create "short hand" for the cost function to be minimized
costFunction = @(p) costAutoencoder(p, thetaSize, Xtrain, parameters);

% Now, costFunction is a function that takes in only one argument (the 
% neural network parameters). Use tic and toc to see how long the training
% takes.
tic
options = struct('display', 'on', 'MaxIter', 400);
[optTheta, optCost] = fmincg(costFunction, theta, options);
toc

% You can visualize what the network has learned by plotting the weights of
% W1 using displayData.
[W1 W2 b1 b2] = theta2params(optTheta, thetaSize);
displayData(W1);

figure;
h = sigmoid(bsxfun(@plus, W1*Xtrain, b1)); %hidden layer
Xrec = sigmoid(bsxfun(@plus, W2*h, b2)); % reconstruction layer
subplot(1,2,1); displayData(Xtrain(:,1:100)'); title('Original input')
subplot(1,2,2); displayData(Xrec(:,1:100)'); title('Reconstructions')

figure; 
imagesc(h); title(['Mean hidden unit activation: ' num2str(mean(mean(h,2)))])

% Use hidden layer for classification
htrain = sigmoid(bsxfun(@plus, W1*Xtrain, b1)); %hidden layer
hval = sigmoid(bsxfun(@plus, W1*Xval, b1)); %hidden layer
htest = sigmoid(bsxfun(@plus, W1*Xtest, b1)); %hidden layer

% With softmax classifier
numClasses = 10;
for lambda = [0 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1]
options = struct('display', 'off', 'maxIter', 400);
theta = trainSoftmax(htrain, ytrain, numClasses, lambda, options);
fprintf('Train Set Accuracy: %f\n', mean(predictSoftmax(theta, htrain, numClasses)==ytrain)*100) ;
fprintf('Validation Set Accuracy: %f\n', mean( predictSoftmax(theta, hval, numClasses)==yval)*100);
fprintf('Test Set Accuracy: %f\n', mean(predictSoftmax(theta, htest, numClasses)==ytest)*100);
end

% With logistic regression classifier
for lambda = [0 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1]
all_theta = trainLogisticReg(h', ytrain', lambda);
htest = sigmoid(bsxfun(@plus, W1*Xtest, b1)); %hidden layer
fprintf('Train Set Accuracy: %f\n', mean(predictLogisticReg(all_theta, htrain')==ytrain')*100);
fprintf('Val Set Accuracy: %f\n', mean(predictLogisticReg(all_theta, hval')==yval')*100);
fprintf('Test Set Accuracy: %f\n', mean(predictLogisticReg(all_theta, htest')==ytest')*100);
end


%% ================== Part 15: Anomaly Detection ===========================
clear all;

% Load the anomaly data. 
load anomalydata

% The data consist of training data X and validation data Xval sampled from 
% a multivariate gaussian distribution. The label vector yval(i)=0 if the 
% ith validation data point is faulty-free and yval(i)=1 if it is a fault.
% We use 1000 training data points, 500 fault-free validation points, and 
% 10 faulty validation points. The data was generated using the following
% code:
% rng('default') %reset the random number generator
% rng(1) % set the seed to 1
% mu = [10,20];
% sigma = [10,5;5,5];
% X = mvnrnd(mu,sigma,1000);
% Xval = [mvnrnd(mu,sigma,500); mvnrnd(mu,sigma*50,10)];
% yval = [zeros(500, 1); ones(10,1)];

% Visualize the example dataset
figure; hold on;
plot(X(:,1), X(:,2),'k+')
plot(Xval(:,1), Xval(:,2),'r+')
plot(Xval(yval==1,1), Xval(yval==1,2), 'ro', 'LineWidth', 2, 'MarkerSize', 10);
legend('Training data', 'Validation data', 'Anomaly')

% Calculate the covariance matrix Sigma and the mean vector mu for the
% training data X. You can use MATALB's functions cov and mean or use the
% definitions from the lecture.
Sigma = cov(X);
mu = mean(X,1)';

% Sigma = zeros(n,n);
% for i=1:n
%     for j = 1:n
%         Sigma(i,j) = 1/(m-1)*sum(sum(bsxfun(@minus, X(:,i), mu(i)).*bsxfun(@minus, X(:,j), mu(j))));
%     end
% end

% Return the density of the multivariate normal at each data point (row) of X
% Complete the code in multivariateGaussian.m and then use visualizeFit to 
% see a contour plot of the fit. 
p = multivariateGaussian(X, mu, Sigma);
figure; hold on;
plot(X(:,1), X(:,2),'k+')
visualizeFit(X, mu, Sigma, 10.^(-20:5:0)');

% Now you will find a good epsilon threshold using a cross-validation set
% probabilities given the estimated Gaussian distribution. Complete the code in
% selectThreshold.m before continuing. You should be able to get a F1-score of 1. 
pval = multivariateGaussian(Xval, mu, Sigma);
[epsilon F1] = selectThreshold(yval, pval);

% Visualize the data and the optimal choice for epsilon.
figure; hold on;
plot(X(:,1), X(:,2),'k+')
plot(Xval(:,1), Xval(:,2),'r+')
plot(Xval(yval==1,1), Xval(yval==1,2), 'ro', 'LineWidth', 2, 'MarkerSize', 10);
visualizeFit([X; Xval], mu, Sigma, epsilon);
legend('Training data', 'Validation data', 'Anomaly'); title(['Epsilon: ' num2str(epsilon)])


%% =========== Part 16: Implementing Collaborative filtering ==============
% In this exercise you will implement the collaborative filtering algorithm.
close all; clear all; clc

% First load the data. The data is a subset of the MovieLens data set which
% contains ratings (1-5) of movies. The full data set contains ratings of 
% 10,000 movies by 72,000 users and can be found at: 
% http://grouplens.org/datasets/movielens/
load ('smallMovieLens.mat'); % Gives Y and R
% This subset only contains 1682 movies by 943 users.
% Y is a 1682x943 matrix, where Y(i,j) is the rating of movie i by user j
% R is a 1682x943 matrix, where R(i,j)=1 if user j gave a rating to movie i

%  From the matrix, we can compute statistics like average rating.
fprintf('Average rating for movie 1 (Toy Story): %f\n', mean(Y(1,R(1,:))));

% Reduce the data set size so the gradient checking is faster.
num_users = 5; 
num_movies = 10; 
num_features = 7;
Xsmall = rand(num_movies, num_features);
Thetasmall = rand(num_users, num_features);
Ysmall = Y(1:num_movies, 1:num_users);
Rsmall = R(1:num_movies, 1:num_users);

% Finish the code in costCollabFilter.m before continuing.
lambda = 0; % Set this to 1 after you have implemented regularization
[J grad] = costCollabFilter([Xsmall(:) ; Thetasmall(:)], Ysmall, Rsmall, ...
    num_users, num_movies, num_features, lambda);
numgrad = checkGradient(@(t) costCollabFilter(t, Ysmall, Rsmall, num_users, ...
    num_movies, num_features, lambda), [Xsmall(:); Thetasmall(:)]);
diff = norm(numgrad-grad)/norm(numgrad+grad) % Should be less than 1e-9

% In this exercise you will predict movie ratings.
close all;

% Load data again
load ('smallMovieLens.mat'); % Gives Y and R

% Read the movie names from the file movie_idx.txt
movieList = loadMovieList();

% Now add your own ratings. You can add or change these ratings if you
% want.
my_ratings = zeros(1682, 1);
my_ratings(1) = 4; % 'Toy Story (1995)'
my_ratings(7) = 3; % 'Twelve Monkeys (1995)'
my_ratings(12)= 5; % 'Usual Suspects, The (1995)'
my_ratings(54) = 4; % 'Outbreak (1995)'
my_ratings(64)= 5; % 'Shawshank Redemption, The (1994)'
my_ratings(66)= 3; % 'While You Were Sleeping (1995)'
my_ratings(69) = 5; % 'Forrest Gump (1994)'
my_ratings(98) = 2; % 'Silence of the Lambs, The (1991)'
my_ratings(183) = 4; % 'Alien (1979)'
my_ratings(226) = 5; % 'Die Hard 2 (1990)'
my_ratings(355)= 5; % 'Sphere (1998)'

%  Add your own ratings to the data matrix
Y = [my_ratings Y];
R = [(my_ratings~=0) R];
[num_movies num_users] = size(Y);

%  Normalize Ratings (not necessary)
%[Ynorm, Ymean] = normalizeRatings(Y, R);

% User set hyperparameters. You can experiment with these if you suspect
% the algortihm is not perform well enough.
num_features = 10;
lambda = 10;

% Set Initial Parameters (Theta, X)
X = randn(num_movies, num_features);
Theta = randn(num_users, num_features);
initial_parameters = [X(:); Theta(:)];

% Train!
options = optimset('GradObj', 'on', 'MaxIter', 100);
theta = fmincg (@(t)(costCollabFilter(t, Y, R, num_users, num_movies, ...
    num_features, lambda)), initial_parameters, options);

% Unfold the returned theta back into X and Theta
X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(theta(num_movies*num_features+1:end), num_users, num_features);

% List the top 10 recommended movies (highest predicted rating) of all the
% movies that you have not yet rated. Compute your recommendations
p = X * Theta';
my_predictions = p(:,1); % + Ymean;

[r, ix] = sort(my_predictions, 'descend');
fprintf('\nTop recommendations for you:\n');
for i=1:10
    j = ix(i);
    fprintf('Predicting rating %.1f for movie %s\n', my_predictions(j), movieList{j});
end


%% ============= Part 17: Convolutional Neural Networks ===================
% Time to implement the convolution and pooling in a CNN. We use random
% test data to verify the implementation. The test data is a 4-D matrix
% with dimensions imageDim x imageDim x numChannels x numImages.

clear all;

imageDim = 8;
filterDim = 4;
numImages = 2;
numFilters = 3;
testMatrix = randn(imageDim, imageDim, 1, numImages);
testFilters = randn(filterDim, filterDim, 1, numFilters);

% Implement and test convolution
convolved = cnnConvolve(testMatrix, testFilters);
for i = 1:1000    
    filter = randi(numFilters);
    image = randi(numImages);
    imageRow = randi(imageDim - filterDim + 1);
    imageCol = randi([1, imageDim - filterDim + 1]);    

    calculatedValue = convolved(imageRow, imageCol, filter, image);
    expectedValue = sum(sum(testFilters(:,:,1,filter).*...
        testMatrix(imageRow:imageRow+filterDim-1,imageCol:imageCol+filterDim-1,1,image)));
    
    if abs(calculatedValue - expectedValue) > 1e-9
        fprintf('Filter Number    : %d\n', filterNum);
        fprintf('Image Number      : %d\n', imageNum);
        fprintf('Image Row         : %d\n', imageRow);
        fprintf('Image Column      : %d\n', imageCol);
        fprintf('Convolved value : %0.5f\n', calculatedValue);
        fprintf('Expected value : %0.5f\n', expectedValue);
        error('Convolved feature does not match expected value');
    else
        fprintf('Implementation of convolution is correct\n');
    end
end

% Implement and test pooling
convolvedDim = 8;
numFilters = 2;
poolDim = 4; % p
poolingmethod = 'mean';
testMatrix = randn(convolvedDim, convolvedDim, 1, numFilters);
expectedMatrix = zeros(2,2,1,numFilters);
for i=1:2
    expectedMatrix(:,:,1,i) = [mean(mean(testMatrix(1:poolDim, 1:poolDim,1,i)))...
        mean(mean(testMatrix(1:poolDim, poolDim+1:end,1,i))); ...
        mean(mean(testMatrix(poolDim+1:end, 1:poolDim,1,i))) ...
        mean(mean(testMatrix(poolDim+1:end, poolDim+1:end,1,i)))];
end
pooled = cnnPool(testMatrix, poolDim, poolingmethod);
if mean(mean(mean(pooled==expectedMatrix)))==1
    fprintf('Implementation of pooling is correct\n');
  else
    error('Pooling layer does not match expected value');
end


clear all;

% We will implement mini-batch gradient descent for a CNN on the MNIST data.
% Copy smallMNIST.mat and any other necessary files from previous labs if 
% you need to.
load smallMNIST
numClasses = 10;

% Need to normalize the data this time because we will use sigmoid
% activation function and learn filters with k-means instead of backprop.
X = zscore(X);

seed = 1;
[Xtrain, Xval, Xtest] = splitData(X, [0.8; 0.1; 0.1], seed);
[ytrain, yval, ytest] = splitData(y, [0.8; 0.1; 0.1], seed);

% Reshape the data to a 4-D matrix with dimensions 
% imageDim x imageDim x numChannels x numImages. MNIST are grey-scale so
% numChannels = 1. 
Xtrain = reshape(Xtrain', 20, 20, 1, []);
Xval = reshape(Xval', 20, 20, 1, []);
Xtest = reshape(Xtest', 20, 20, 1, []);
ytrain = ytrain'; yval = yval'; ytest = ytest';

% We can look at the first image with
imagesc(Xtrain(:,:,:,1)); colormap('gray')

% User parameter
filterDim = 9;
poolDim = 2;
poolingmethod = 'mean';

% We will learn the filters using k-means. First we need to randomly
% extract patches with size filterDim x filterDim from the large images.
numPatches = 10000;
patches = samplePatches(Xtrain, filterDim, numPatches);

% Next we train 50 filters with k-means.
numFilters = 50;
filters = run_kmeans(patches', numFilters, 100, true)';

% You can also learn filters with an auto-encoder with something like this. (not tested)
% parameters.lambda = 1e-4;
% parameters.beta = 0.03;
% [theta thetaSize] = initAEParameters(filterDim^2, numFilters);
% options = struct('MaxIter', 400);
% [optTheta, cost] = fmincg( @(p) costAutoencoder(p, thetaSize, patches, parameters), theta, options);
% [W1 W2 b1 b2] = theta2params(theta, thetaSize);
% filters = W1';

% We can display the 100 first extracted patches and the 50 learned filters
% with the following code.
figure; displayData(patches(:,1:100)')
figure; displayData(filters(:,1:numFilters)')

% Feedforward train data
filters = reshape(filters, filterDim, filterDim, 1, []);
convolved = cnnConvolve(Xtrain, filters);
convolved = sigmoid(convolved);
pooledtrain = cnnPool(convolved, poolDim, poolingmethod);
pooledtrain = reshape(pooledtrain, [], size(pooledtrain,4)); %reshape for softmax

% Feedforward validation data
convolvedval = cnnConvolve(Xval, filters);
convolvedval = sigmoid(convolvedval);
pooledval = cnnPool(convolvedval, poolDim, poolingmethod);
pooledval = reshape(pooledval, [], size(pooledval,4));

% Train softmax classifier with mini-batch gradient descent. Finish the
% incomplete code in minibatchGD.m before continuing. 
parameters.lambda = 1e-4;
parameters.numClasses = numClasses;
parameters.numepochs = 400;
parameters.batchsize = 200;
parameters.learningRate = 0.01;
parameters.useDecayLearningRate = true;
parameters.useMomentum = true;
[optTheta Jtrain Jval] = minibatchGD(pooledtrain, ytrain, parameters, pooledval, yval);

% Plot the classification error for the train and validation data.
figure; hold on;
plot(100*(1-Jtrain), 'Color', 'k')
plot(100*(1-Jval), 'Color', 'r')
legend('Jtrain', 'Jval')

% Sometimes can be easier to see if the classification accuracy on the
% validation set is decreasing or not if we use averaging.
numAvgpoints = 10;
figure; hold on;
plot(1-mean(reshape(Jtrain, numAvgpoints,[])), 'Color', 'k')
plot(1-mean(reshape(Jval, numAvgpoints,[])), 'Color', 'r')
legend('Jtrain', 'Jval')

% Feedforward test data and classify
convolvedtest = cnnConvolve(Xtest, filters);
convolvedtest = sigmoid(convolvedtest);
pooledtest = cnnPool(convolvedtest, poolDim, poolingmethod);
pooledtest = reshape(pooledtest, [], size(pooledtest,4));
predtest = predictSoftmax(optTheta, pooledtest, numClasses);
testAccuracy = mean(predtest == ytest);
fprintf('Test accuracy: %0.2f\n', testAccuracy*100)

% You can print a confusion matrix with the following code
confmat(predtest, ytest)




