function [J, grad] = costLinearRegression(theta, X, y)
% Compute the cost (J) and gradient (grad) for linear regression with 
% squared error cost function for input data (X), target data (y), and 
% parameters (theta).

m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Compute the squared error cost.
% HINT: Start with calculating the squared error cost for one training
% example at a time and sum the total cost in a for-loop. Then vectorize 
% the code when you have checked that you have the correct cost and 
% gradient.
%J = ...
J = 1/(2*m)*sum((X*theta-y).^2);
% =========================================================================

% ====================== YOUR CODE HERE ======================
% Compute the gradient. 
% HINT: First compute the gradient for the first parameter for the first 
% training example. Then compute the gradient for the second parameter for
% the first training example. Then do the same for all training examples
% and sum up the gradients. The variable grad should be a vector of same
% length as the number of parameters (theta).
%grad = ...
grad=(1/m)*((theta'*X'-y')*X);
% =========================================================================

grad = grad(:); % Turn grad into a n+1-by-1 vector.


end
