function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(size(X, 1), 1) X];
x=sigmoid(X*Theta1');

activation_layer2= [ones(size(x, 1), 1) x];

activated_output_layer=sigmoid(activation_layer2*Theta2');


size(activated_output_layer)

y1=zeros(size(activated_output_layer));
regularized=0;
regularized_part1=0;
regularized_part2=0;

for j=1:size(Theta1,1)
    for k1=2:size(Theta1,2)
        regularized_part1=regularized_part1+(lambda/(2*m))*(Theta1(j,k1).^2);
        
    end
    
end

for j=1:size(Theta2,1)
    for k2=2:size(Theta2,2)
        regularized_part2=regularized_part2+(lambda/(2*m))*(Theta2(j,k2).^2);
        
    end
    
end

regularized=regularized_part1+regularized_part2;

for iter=1:m
    y1(iter,y(iter))=1;
    
end    

sum1=0;
for i=1:m
    for k=1:num_labels
         sum1=sum1+(y1(i,k)'*log(activated_output_layer(i,k))+((1-y1(i,k)')*log(1-activated_output_layer(i,k))));
         J=(-1/m)*sum1;
        
    end
 
end

J=J+regularized;


d3=activated_output_layer-y1;


d2=(d3*Theta2);
d2(:,1)=[];
d2=d2.*sigmoidGradient(X*Theta1');

D1=zeros(size(Theta1));
D2=zeros(size(Theta2));
D1=D1+d2'*X;
D2=D2+d3'*activation_layer2;

regularizing_theta1=(lambda/m)*Theta1;
regularizing_theta1(:,1)=[];
regularizing_theta1=[zeros(size(regularizing_theta1, 1), 1) regularizing_theta1];
regularizing_theta2=(lambda/m)*Theta2;
regularizing_theta2(:,1)=[];
regularizing_theta2=[zeros(size(regularizing_theta2, 1), 1) regularizing_theta2];
size(regularizing_theta1);
size(regularizing_theta2);
size(D1);
Theta1_grad=(1/m)*D1+regularizing_theta1;
Theta2_grad=(1/m)*D2+regularizing_theta2;

    







% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
