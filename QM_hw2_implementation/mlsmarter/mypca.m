function [coefs, scores, variances] = pca(X)

[m, n] = size(X);
[coefs, variances, temp] = svd(1/m*(X'*X));
variances = diag(variances);

% Uncomment this to make it more like princomp (but will still not be
% exactly the same). This will flip the sign so the greatest value of each 
% column in coefs is positive.
% [~,maxind] = max(abs(coefs),[],1);
% colsign = sign(coefs(maxind + (0:n:(n-1)*n)));
% coefs = bsxfun(@times,coefs,colsign);

scores = zeros(size(X));

% ====================== YOUR CODE HERE ======================
% Compute the projection of the data using the component
% coefficients. For the i-th example X(i,:), the projection on to the k-th 
% eigenvector is given as follows: scores(i,k) = X(i, :) * coefs(:, k);
% scores = ...

% for i = 1:size(X,1)
%     for k=1:n
%         scores(i,k) = X(i, :) * coefs(:, k);
%     end
% end
scores = X*coefs;

end
