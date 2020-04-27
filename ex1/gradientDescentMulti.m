function [theta, J_history] = gradientDescentMulti(X_norm, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    predictions=X_norm*transpose(theta);
    for iter1 =1:size(theta,2)
        
    	theta(iter1)=theta(iter1)-(alpha*(1/m)*sum(transpose(predictions-y)*X_norm(:,iter1)));
        
    end
    % ============================================================
    
    % Save the cost J in every iteration
    J=computeCostMulti(X_norm, y, theta);
    J_history(iter)=J;
    J_history(iter)










end

end
