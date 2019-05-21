function p = predictLogisticReg(all_theta, X)

m = size(X, 1);

% You need to return the following variables correctly 
p = zeros(m, 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Make predictions using the learned logistic regression parameters 
% (one-vs-all). First calculate the hypothesis for each of the learned 
% model with h_all = sigmoid(all_theta'*X'). Then set p(i) = k to the kth 
% column that has the maximum value in row i.
%
% This code can be done all vectorized using the max function. In 
% particular, the max function can return the index of the max 
% element as its second output argument. If your examples are in 
% rows then you can use [~, ind] = max(A, [], 1) to obtain the index for
% the max of each column.

[~,p] = max(sigmoid(all_theta'*X'),[],1);

% =========================================================================

% Unroll to make sure p is a row vector
p = p(:);

end

