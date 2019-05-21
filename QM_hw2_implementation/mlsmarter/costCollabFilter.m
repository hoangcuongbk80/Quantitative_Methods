function [J, grad] = costCollabFilter(params, Y, R, num_users, num_movies, num_features, lambda)
% X     - num_movies x num_features
% Theta - num_users  x num_features
% Y     - num_movies x num_users (ratings 1-5)
% R     - num_movies x num_users (rated 0-1)

% Unfold the X and Theta matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), num_users, num_features);

% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Compute the cost function and gradient for collaborative filtering. Do 
% this step-by-step. 
%
% Step 1: Compute J non-vectorized. This can be done with 2 for-loops:
% Jmat=zeros(size(R));
% for i=1:size(R,1)
%     for j=1:size(R,2)
%         if R(i,j)==1
%             Jmat(i,j) = (Theta(j,:)*X(i,:)' - Y(i,j)).^2;
%         end
%     end
% end
% J = 0.5*sum(sum(Jmat));
%
% Step 2: Compute X_grad and Theta_grad non-vectorized. Then check
% gradients. An example of X_grad is provided below. Do similarly for
% Theta_grad (Remember to sum over the number of movies instead of the
% number of users).
% for k=1:size(X,2)
%     for i=1:num_movies
%         temp=0;
%         for j=1:num_users
%             if R(i,j)==1
%                 temp = temp + (Theta(j,:)*X(i,:)' - Y(i,j))*Theta(j,k);
%             end
%         end
%         X_grad(i,k) = temp;
%     end
% end
%
% Step 3: Add regularization terms to cost function and the derivatives.
% Check gradients. Below is the added regularization for X. Add the
% regularization for Theta as well. Set lambda = 1 in lab4.m.
% J = J + lambda/2*sum(sum(X.^2)) + ...
% X_grad = X_grad + lambda*X;
% Theta_grad = ...
%
% Step 4: (Optional) Vectorize J. Check gradients.
%
% You can use the R matrix to set selected entries to 0 with A.*R.. Since R 
% only has elements with values either 0 or 1, this has the effect of 
% setting the elements of A to 0 only when the corresponding value in R is 
% 0. 
% 
% Step 5: (Optional) Vectorize gradients. Make use of the R matrix.

% 
% Jmat=zeros(size(R));
% for i=1:size(R,1)
%     for j=1:size(R,2)
%         if R(i,j)==1
%             Jmat(i,j) = (Theta(j,:)*X(i,:)' - Y(i,j)).^2;
%         end
%     end
% end
% J = 0.5*sum(sum(Jmat));
% 
% for k=1:size(X,2)
%     for i=1:num_movies
%         temp=0;
%         for j=1:num_users
%             if R(i,j)==1
%                 temp = temp + (Theta(j,:)*X(i,:)' - Y(i,j))*Theta(j,k);
%             end
%         end
%         X_grad(i,k) = temp;
%     end
% end
% 
% for k=1:size(X,2)
%     for j=1:num_users
%         temp=0;
%         for i=1:num_movies
%             if R(i,j)==1
%                 temp = temp + (Theta(j,:)*X(i,:)' - Y(i,j))*X(i,k);
%             end
%         end
%         Theta_grad(j,k) = temp;
%     end
% end
% 
% With added regularization
% J = J + lambda/2*sum(sum(X.^2)) + lambda/2*sum(sum(Theta.^2));
% X_grad = X_grad + lambda*X;
% Theta_grad = Theta_grad + lambda*Theta;

% Vectorized and regularized
J = 0.5*sum(sum(((X*Theta' - Y).^2).*R)) + lambda/2*sum(sum(X.^2)) + lambda/2*sum(sum(Theta.^2));
X_grad = ((X*Theta' - Y).*R)*Theta + lambda*X;
Theta_grad = ((X*Theta' - Y).*R)'*X + lambda*Theta;

% =============================================================

grad = [X_grad(:); Theta_grad(:)]; % Unroll both X_grad and Theta_grad

end
