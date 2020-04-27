function J = computeCostMulti(X_norm, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

predictions=X_norm*transpose(theta);
error=predictions-y;
squared_error=error.^2;

J=1*0.5*(1/m)*sum(squared_error);




% =========================================================================

end
