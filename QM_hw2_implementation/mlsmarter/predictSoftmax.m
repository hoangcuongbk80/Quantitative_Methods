function pred = predictSoftmax(theta, X, numClasses)

% theta - model trained using trainSoftmax
% X - the n x m input matrix, where each column X(:, i) corresponds to a 
% single test set
%
% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
% Unroll the parameters from theta
theta = reshape(theta, numClasses, size(X,1));
pred = zeros(1, size(X,2));

% ---------- YOUR CODE HERE --------------------------------------
% Compute pred using theta and X assuming that the labels start from 1. You
% can use the code that calculates h from costSoftmax.m and use 
% [maxvalue, maxrow] = max(h, [], 1); where maxvalue(i) is the maximum 
% value of column i and maxrow(i) is the row that has the maximum value in 
% column i.

h = theta*X;
h = bsxfun(@minus, h, max(h, [], 1)); % Preventing overflows 
h = exp(h);
h = bsxfun(@rdivide, h, sum(h));
[~, pred] = max(h,[],1);

% ---------------------------------------------------------------------

% Make sure pred is a column vector
pred = pred(:)';

end

