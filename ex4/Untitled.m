clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 7;  % 7 features used
hidden_layer_size = 14;   % 25 hidden units
num_labels = 1;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)
data =readtable('train.csv');
data1=readtable('test.csv');

data=rmmissing(data);
data(:,4)=[];
data1(:,3)=[];
data(:,8)=[];
data1(:,7)=[];
data(:,9)=[];
data1(:,8)=[];
data(:,9)=[];
data1(:,8)=[];

avgAge = nanmean(data.Age)             % get average age
data.Age(isnan(data.Age)) = avgAge;   % replace NaN with the average
data1.Age(isnan(data1.Age)) = avgAge;     % replace NaN with the average
data.Age;
data1;
X=data.Age;
mu=mean(X);
sigma=std(X);
data.Age=(X-mu)/sigma;
X1=data1.Age; 
data1.Age=(X1-mu)/sigma;


X=data.Fare;


mu=mean(data.Fare);
sigma=std(data.Fare);
data.Fare=(X-mu)/sigma;
data;
X1=data1.Fare;
data1.Fare(isnan(data1.Fare))=mu;

X=data.Sex;

X2={1;0};

for iter=1:size(X,1)
    if(strcmp(char(X(iter)),'female'))
        X(iter)=X2(1);
        
   elseif(strcmp(char(X(iter)),'male'))
         X(iter)=X2(2);
    end
end
X1=cell2mat(X);

X=data1.Sex;
for iter=1:size(X,1)
    if(strcmp(char(X(iter)),'female'))
        X(iter)=X2(1);
        
   elseif(strcmp(char(X(iter)),'male'))
         X(iter)=X2(2);
    end
end
X3=cell2mat(X);

data.Sex=X1;
data1.Sex=X3;
y=data.('Survived');
data.('Survived')=[];
X=table2array(data);
X;
data1;
%X = mapFeature(X(:,4), X(:,7));
X;

m = size(X, 1);

fprintf('\nLoading Saved Neural Network Parameters ...\n')

% Load the weights into variables Theta1 and Theta2
load('ex4weights.mat');
Theta1=Theta1(1:25,1:8);
Theta2=Theta2(1:26);
% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];

fprintf('\nFeedforward Using Neural Network ...\n')

% Weight regularization parameter (we set this to 0 here).
lambda = 0;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

fprintf(['Cost at parameters (loaded from ex4weights): %f '...
         '\n(this value should be about 0.287629)\n'], J);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

fprintf('\nChecking Backpropagation... \n');

%  Check gradients by running checkNNGradients
% checkNNGradients;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

fprintf('\nChecking Backpropagation (w/ Regularization) ... \n')

%  Check gradients by running checkNNGradients
lambda = 3;
% checkNNGradients(lambda);

% Also output the costFunction debugging values
debug_J  = nnCostFunction(nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, X, y, lambda);

fprintf(['\n\nCost at (fixed) debugging parameters (w/ lambda = %f): %f ' ...
         '\n(for lambda = 3, this value should be about 0.576051)\n\n'], lambda, debug_J);

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 100);

%  You should also try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);



% Set regularization parameter lambda to 1 (you should vary this)
% lambda = .02;
% 
% p = predict(theta, X);
% 
% fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
% 
% 
% Y=table2array(data1);
% p = predict(theta, Y);







