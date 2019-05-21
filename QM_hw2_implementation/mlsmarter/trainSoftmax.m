function theta = trainSoftmax(X, y, numClasses, lambda, options)
%softmaxTrain Train a softmax model with the given data and labels. Returns 
% the wight matrix theta.
%
% X: an n by m matrix containing the input data, such that X(:, c) is the 
% cth input
% y: m by 1 matrix containing the class labels for the corresponding 
% inputs. labels(c) is the class label for the cth input
% options: (optional) Can change the number of iterations

if ~exist('options', 'var')
    options = struct;
end

if ~isfield(options, 'maxIter')
    options.MaxIter = 400;
end

options.Display = 'On';

% initialize parameters
initial_theta = reshape(0.005 * randn(numClasses, size(X, 1)), [], 1);

% ====================== YOUR CODE HERE ======================
% Use any choice of optimization solver to calculate theta (minFunc,
% fmincg, fminunc, or gradientDescent.m from lab 1).

% Use minFunc to minimize the function
%theta = minFunc( @(p) costSoftmax(p, X, y, numClasses, lambda), initial_theta, options);
%theta = fminunc( @(p) costSoftmax(p, X, y, numClasses, lambda), initial_theta, options);
%theta = fminsearch( @(p) costSoftmax(p, X, y, numClasses, lambda), initial_theta, options);
theta = fmincg( @(p) costSoftmax(p, X, y, numClasses, lambda), initial_theta, options);
%[x, fval, info, output] = fminbnd (fun, a, b, options)

% ============================================================

% Unroll theta
theta = theta(:);
                          
end                          
