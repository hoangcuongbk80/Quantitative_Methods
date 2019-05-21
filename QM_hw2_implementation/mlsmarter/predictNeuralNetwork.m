function p = predictNeuralNetwork(theta, thetaSize, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 2);

% Reshape theta to the network parameters
[W1 W2 b1 b2] = theta2params(theta, thetaSize);

% You need to return the following variables correctly 
p = zeros(1, m);

% ====================== YOUR CODE HERE ======================
% Complete the following code to make predictions using your learned neural 
% network. You should set p to a vector containing labels between 1 to 
% num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 1) to obtain the max for each row.

z2 = bsxfun(@plus, W1 * X, b1);
a2 = sigmoid(z2);
z3 = bsxfun(@plus, W2 * a2, b2);
a3 = sigmoid(z3);

[~, p] = max(a3,[],1);

% =========================================================================

% Make sure pred is a column vector
p = p(:)';

end
