function theta = normalEqn(X, y)
%Computes the closed-form solution to linear regression.

% ====================== YOUR CODE HERE ======================
% Use the MATLAB command pinv to calculate the inverse. Use ' for
% transpose.
%theta = ...
theta = pinv(X'*X)*X'*y;
% ============================================================

end
