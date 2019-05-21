function [J, grad] = costAutoencoder(theta, thetaSize, X, parameters)
% Returns cost and gradient vector for a 3-layered Auto-encoder.
% theta - A column vector of the parameters W,b. The structure of theta is:
% theta = [W1(:); W2(:); b1(:) b2(:)]
% thetaSize - A 4x2 matrix where thetaSize(1,1) is the number of rows
% of W1 and thetaSize(1,2) is the number of columns in W1. thetaSize(2,:)
% is the number of rows and columns of W2 etc.
% X - the n x m input matrix, where each column X(:, i) corresponds to one
% training example
% parameters - a structure of all the parameters

% Useful parameters
m = size(X, 2);

% Reshape theta to the network parameters
[W1 W2 b1 b2] = theta2params(theta, thetaSize);

% Get the parameters
lambda = parameters.lambda; % Weight decay penalty parameter
beta = parameters.beta; % sparsity penalty parameter
p = 0.05; % sparsity activation parameter rho. This can actually also be 
% considered a tunable parameter but we set this to 0.05.

% You should compute the following variables
J = 0;
gradW1 = zeros(size(W1));
gradW2 = zeros(size(W2));
gradb1 = zeros(size(b1));
gradb2 = zeros(size(b2));

% ---------- YOUR CODE HERE --------------------------------------
% Compute the cost and gradient for auto-encoder. If you have completed
% costNeuralNetwork.m this is very easy. Copy the code from 
% costNeuralNetwork and just change the calculation of sqerrterm to:
% sqerrterm = mean(0.5*sum((a3-a1).^2,1));
% And delta3 to:
% delta3 = -(a1-a3).*sigmoidGradient(z3);


z2 = bsxfun(@plus, W1 * X, b1);
a2 = sigmoid(z2);
z3 = bsxfun(@plus, W2 * a2, b2);
%a3 = sigmoid(z3);
a3 = z3; % For linear decoder

%sqerrterm = -1/m*sum(sum(y.*log(a3)+(1-y).*log(1-a3)));
sqerrterm = mean(0.5*sum((a3-X).^2,1));

weightdecayterm = lambda/2*(sum(sum(W1.^2))+sum(sum(W2.^2)));

pj = mean(a2,2);
sparsitypenalty = beta*sum(p*log(p./pj) + (1-p)*log((1-p)./(1-pj)));

J = sqerrterm + weightdecayterm + sparsitypenalty;

% delta3 = -(y-a3); % For cross-entropy error
%delta3 = -(X-a3).*sigmoidGradient(z3); % For squared error
delta3 = -(X-a3); % For linear decodeer
delta2 = bsxfun(@plus, W2'*delta3, beta*(-p./pj + (1-p)./(1-pj))).*sigmoidGradient(z2);
gradW2 = delta3*a2'/m + lambda*W2;
gradW1 = delta2*X'/m + lambda*W1;
gradb2 = sum(delta3,2)/m;
gradb1 = sum(delta2,2)/m;

% ------------------------------------------------------------------

% Unroll the gradient matrices into a vector for the gradient method
grad =  [gradW1(:); gradW2(:); gradb1(:); gradb2(:)];

end

% Some helping functions that you might use. These can only be called 
% inside this function (costNeuralNetwork.m)
function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
    z = sigmoid(z);
    g = z.*(1-z);
end

function sigm = sigmoid(z)
% SIGMOID return the output from the sigmoid function with input z
sigm = 1./(1 + exp(-z));
end


