function p = multivariateGaussian(X, mu, Sigma)
% Computes the probability density function of the multivariate gaussian 
% distribution.
% X is the m x n data matrix with m data points and n features.

% Get the number of training points m and number of features n.
[m n] = size(X);

if (size(Sigma, 2) == 1) || (size(Sigma, 1) == 1)
    Sigma = diag(Sigma);
end

% Return the following variable
p = zeros(m,1);

% ====================== YOUR CODE HERE ======================
% Compute the multivariate gaussian probability with covariance matrix Sigma
% and mean vector mu for each data point in X. Use pinv(A) to calculate the
% inverse of A and det(A) to calculate the determinant of A. 

p = (2 * pi) ^ (- n / 2) * det(Sigma) ^ (-0.5) * ...
    exp(-0.5 * sum(bsxfun(@times, bsxfun(@minus, X, mu') * pinv(Sigma), bsxfun(@minus, X, mu')), 2));

% const = 1/sqrt((2*pi)^n*det(Sigma));
% invSigma = pinv(Sigma);
% p = zeros(m,1);
% for i=1:m
%     p(i) = const * exp(-0.5*(X(i,:)'-mu)'*invSigma*(X(i,:)'-mu));
% end

% ============================================================

% Unroll p
p = p(:);



end