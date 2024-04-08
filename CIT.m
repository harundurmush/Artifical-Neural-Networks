clc;
clear all;
close all;
%%
% Define input matrix (each column represents an input vector)
X = [1 1 1 1 0 0 1 1 1;  % Example input for class C
     0 1 0 0 1 0 0 1 0;  % Example input for class I
     1 1 1 0 1 0 0 1 0]; % Example input for class T

% Define target matrix (each column represents a target vector for each class)
T = [1 0 0;  % Target for class C
     0 1 0;  % Target for class I
     0 0 1]; % Target for class T

% Create a feedforward network with 2 hidden layers of 10 neurons each
% This is an arbitrary choice and would be adjusted based on the specific problem
net = feedforwardnet([10 10]);

% Configure the network to fit the input and target data
net = configure(net, X, T);

% Split the data into training and testing sets
[trainInd, valInd, testInd] = divideind(size(X, 2), 1:3, 4:6, 7:9); % Adjust this based on your data split requirements

X_train = X(:, trainInd);
T_train = T(:, trainInd);

% Train the network
[net, tr] = train(net, X_train, T_train);

% Test the network with training data
Y_train = net(X_train);
trainPerformance = perform(net, T_train, Y_train);

% Display training performance
disp(['Training Performance: ' num2str(trainPerformance)]);


% Reshape newInput to match the expected input size (1x9 row vector)
% Test the network with new input
newInput = [0 0 1 1 0 0 1 1 1]; % New input for testing

% Reshape newInput to match the expected input size (3x1 column vector)
newInput = reshape(newInput, [], 1);

newOutput = net(newInput);

% Display the output
disp('Network Output:');
disp(newOutput);

% Convert continuous outputs to binary to determine the class
% This is necessary if your network outputs continuous values that need to be thresholded
predictedClass = newOutput > 0.5;

% Display the predicted class
disp('Predicted Class:');
disp(predictedClass);

% For validation (if you have validation data)
valInputs = []; % Your validation input data here
valTargets = []; % Your validation target data here

% Test the network with the validation set
valOutputs = net(valInputs);
valPerformance = perform(net, valTargets, valOutputs);

% Display validation performance
disp(['Validation Performance: ' num2str(valPerformance)]);

% Plot confusion matrix for the validation set
plotconfusion(valTargets, valOutputs);
