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
X

initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1 (you should vary this)
lambda = .02;

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
p = predict(theta, X);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
data1;
Y=table2array(data1);
p = predict(theta, Y);
% 
PassengerId = data1.PassengerId;             % extract Passenger Ids
Survived = predict(theta, Y);             % generate response variable
           % convert to double
submission = table(PassengerId,Survived);   % combine them into a table
disp(submission(1:5,:))                     % preview the table
writetable(submission,'submission.csv')     % write to a CSV file


        
    


