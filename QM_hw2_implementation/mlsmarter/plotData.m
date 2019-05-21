function plotData(X, y)
% Plots the 2-dimensional matrix X. Each data point of the same class has
% the same color/marker. X is a m-by-2 matrix and y is a n-by-1 vector.

K = length(unique(y)); %number of classes

% ====================== YOUR CODE HERE ===================================
% Plot the data in X with different colors/markers according to the label
% vector y.

colors = {'r.' 'k.' 'c.' 'b.' 'y.' 'g.' 'y.' 'bx' 'gx' 'rx' 'cx' 'mx' 'yx' 'kx'};
hold on;
for i=1:K
    plot(X(y==i,1), X(y==i,2), colors{i});
end
% =========================================================================

end