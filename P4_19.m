clc;
clear all;
close all;

%%

% Define the training patterns and targets based on the problem statement
% Each pattern is a point (x1, x2) and the corresponding target is the class
% The targets are given in bipolar format

% Training patterns (as given in Figure P4.19)
input_patterns = [
    1/2, 1/8;
    1/4, 3/4;  % Class 1
    1/2, 1/2;  % Class 2
    3/4, 1/4;  % Class 2
    3/4, 1/8;  % x
    3/4, 1/2;  % x
    1/4, 1/4;  % Class 3
    1/4, 1/2;  % x
    1/2, 3/4;  % x
];

% Targets in bipolar format
targets = [
    -1, 1, -1;  % Class 1
    -1, 1, -1;  % Class 1
    -1, -1, 1;  % Class 2
    -1, 1, -1;  % Class 2
    1, -1, -1;  % 5
    -1, -1, 1;  % 6
    1, -1, -1;  % 7
    -1, -1, 1;  % Class 3
    1, -1, -1;  % Class 3
];

% Network architecture
I = 2;  % Two input neurons
J = 8;  % Number of hidden neurons (adjustable based on problem complexity)
K = 3;  % Three output neurons

% Learning rate
eta = 0.04;

% Mean Squared Error goal (E_rms)
epsilon = 0;

% Train the network
[W1, b1, W2, b2] = train_network(input_patterns, targets, I, J, K, eta, epsilon);
%% buraya gir
example_pattern = [1/4,7/8]; 
predicted_output = test_network(W1, b1, W2, b2, example_pattern);
disp('Predicted output for the example pattern:');
disp(predicted_output);
% Function to train the network (based on P4.15 guidelines)
function [W1, b1, W2, b2] = train_network(input_patterns, targets, I, J, K, eta, epsilon)
    % Initialize weights and biases
    W1 = rand(J, I) * 2 - 1; % Weights from input to hidden layer
    b1 = rand(J, 1) * 2 - 1; % Biases for hidden layer
    W2 = rand(K, J) * 2 - 1; % Weights from hidden to output layer
    b2 = rand(K, 1) * 2 - 1; % Biases for output layer

    max_epochs = 100000; % Define the maximum number of epochs
    epoch = 0;
    mean_sq_error = inf;
    while mean_sq_error > epsilon && epoch < max_epochs
        mean_sq_error = 0;
        for p = 1:size(input_patterns, 1)
            % Forward pass
            input = input_patterns(p, :)';
            target = targets(p, :)';
            
            % Hidden layer activations
            z_in = W1 * input + b1;
            z = tanh(z_in);  % Bipolar sigmoid activation
            
            % Output layer activations
            y_in = W2 * z + b2;
            y = tanh(y_in);  % Bipolar sigmoid activation
            
            % Error calculation
            error = target - y;
            mean_sq_error = mean_sq_error + sum(error .^ 2);
            
            % Backpropagation of error
            delta_k = error .* (1 - y .^ 2);
            delta_j = (W2' * delta_k) .* (1 - z .^ 2);
            
            % Update weights and biases
            W2 = W2 + eta * (delta_k * z');
            b2 = b2 + eta * delta_k;
            W1 = W1 + eta * (delta_j * input');
            b1 = b1 + eta * delta_j;
        end
        
        % Mean square error for epoch
        mean_sq_error = mean_sq_error / size(input_patterns, 1);
        epoch = epoch + 1;
    end
    
    fprintf('Training completed in %d epochs with MSE: %f\n', epoch, mean_sq_error);
end

% After training, use the network to predict class for new patterns
% Here we use an example pattern [0.25, 0.75] which should be close to class 1


% Function to test the network
function output = test_network(W1, b1, W2, b2, pattern)
    z = tanh(W1 * pattern' + b1);
    output = softmax(W2 * z + b2);
end
