function [J, grad] = costLogisticRegression(theta, X, y, lambda)
% Compute cost and gradient for logistic regression.

if nargin<4
    lambda = 0;
end

m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Compute the cost (J) and partial derivatives (grad) of a particular 
% choice of theta. Make use the function sigmoid that you wrote earlier.
% J = ...
% grad = ...

h=sigmoid(X*theta);
J = -1/m*sum((y.*log(h) + (1-y).*log(1-h))) + lambda/2*sum(theta(2:end).^2);
grad = 1/m*((h-y)' * X)' + lambda/m*[0; theta(2:end)];

% =============================================================

grad = grad(:);

end
