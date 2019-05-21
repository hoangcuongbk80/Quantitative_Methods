function visualizeFit(X, mu, Sigma, epsilon)
%VISUALIZEFIT Visualize the dataset and its estimated distribution.
%   VISUALIZEFIT(X, p, mu, sigma2) This visualization shows you the
%   probability density function of the Gaussian distribution. Each example
%   has a location (x1, x2) that depends on its feature values.
%

[X1,X2] = meshgrid(min(min(X)):.5:max(max(X)));
Z = multivariateGaussian([X1(:) X2(:)], mu, Sigma);
Z = reshape(Z,size(X1));

% plot(X(:, 1), X(:, 2), 'k+'); hold on;
% Do not plot if there are infinities
if (sum(isinf(Z)) == 0)
    contour(X1, X2, Z);
end
hold off;
%axis tight;

end
