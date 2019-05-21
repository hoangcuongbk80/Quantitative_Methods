function [coefs, scores, variances, t2] = PlotPCA(X, y)
% Plots the PCA of the data in X. The labels y colors each group with a
% different color.

% ====================== YOUR CODE HERE ===================================
% Calculate the coefficients, scores, and variances for the data matrix X.
% Use the label vector y to plot each data point with a different
% color/marker. Set the xlabel and ylabel to the total variability 
% explained by the first and second principal component respectively. (Use
% the 3rd output argument of princomp for this).

[coefs, scores, variances, t2] = princomp(X);
plotData(scores, y)
varp = 100*variances/sum(variances);
xlabel(['1st P.C. (' num2str(varp(1), '%0.2f') '%)'])
ylabel(['2nd P.C. (' num2str(varp(2), '%0.2f') '%)'])
axis tight;

% =========================================================================

end

