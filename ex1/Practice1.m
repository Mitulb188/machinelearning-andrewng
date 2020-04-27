data = load('ex1data2.txt');
X=data(:,1:2);
no_of_features=size(X,2);
y=data(:,3);
m = length(y);
[X_norm, mu, sigma] = featureNormalize(X);
% Add intercept term to X
X_norm = [ones(m, 1) X_norm];
theta=zeros(1,no_of_features+1);
%J = computeCostMulti(X_norm, y, theta);
% Run gradient descent
% Choose some alpha value
alpha = 0.1;
num_iters = 400;
size(X_norm)
% Init Theta and Run Gradient Descent 

[theta,J_history] = gradientDescentMulti(X_norm, y, theta, alpha, num_iters);

plot(J_history)

 %Display gradient descent's result
fprintf('Theta computed from gradient descent:\n%f,\n%f,\n%f',theta(1),theta(2),theta(3))
normalizedPredictionValues1=(1650-mu(1))/sigma(1);
normalizedPredictionValues2=(3-mu(2))/sigma(2);
predict=[1 normalizedPredictionValues1 normalizedPredictionValues2] * transpose(theta)
fprintf('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f', predict);

