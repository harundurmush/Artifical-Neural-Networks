clc;
clear all;
close all;

%%

% Define the function to approximate
h = @(x) 0.8 * sin(pi * x);

% Training data: 21 points uniformly covering the range [-1, 1]
x_train = linspace(-1, 1, 21)';
y_train = h(x_train);

% Network parameters
J = 10;            % Number of hidden neurons (can be adjusted)
eta = 0.4;         % Learning rate
%% buraya gir
max_epochs = 10000; % Maximum number of epochs for training
epsilon = 0.002089;     % Mean squared error goal

% Train the network
% Train the network
[W1, b1, W2, b2] = train_approximation_network(x_train, y_train, J, eta, max_epochs, epsilon);

% Define the test points: new data points where we want to evaluate the trained network
x_test = linspace(-1, 1, 100)';  % More points for a smoother plot
y_test = h(x_test);              % Actual function values

% Get network's predictions
y_pred = test_trained_network(W1, b1, W2, b2, x_test);

% Plot the results
figure;
plot(x_test, y_test, 'r--', 'LineWidth', 2); % Actual function
hold on;
plot(x_test, y_pred, 'b-', 'LineWidth', 2); % Network approximation
title('Function Approximation with Neural Network');
xlabel('x');
ylabel('h(x) and o(x)');
legend('Actual function h(x)', 'Neural network o(x)', 'Location', 'Best');
grid on;
hold off;

% Display the actual values and predicted values in a table
disp(table(x_test, y_test, y_pred, 'VariableNames', {'x', 'Actual_h_x', 'Predicted_o_x'}));





function y_pred = test_trained_network(W1, b1, W2, b2, input_patterns)
    z = tanh(W1 * input_patterns' + b1);
    y_pred = tanh(W2 * z + b2)';
end

function [W1, b1, W2, b2] = train_approximation_network(input_patterns, targets, J, eta, max_epochs, epsilon)
    % Input and output size
    I = size(input_patterns, 2);  % Input layer size (1 for this problem)
    K = size(targets, 2);         % Output layer size (1 for this problem)
    
    % Initialize weights and biases
    W1 = rand(J, I) * 2 - 1;      % Input to Hidden weights
    b1 = rand(J, 1) * 2 - 1;      % Hidden layer biases
    W2 = rand(K, J) * 2 - 1;      % Hidden to Output weights
    b2 = rand(K, 1) * 2 - 1;      % Output layer biases

    % Training loop
    for epoch = 1:max_epochs
        % Initialize mean squared error
        mean_sq_error = 0;
        % Loop over all training patterns
        for p = 1:size(input_patterns, 1)
            % Forward pass
            input = input_patterns(p, :)';
            target = targets(p, :)';
            z = tanh(W1 * input + b1);
            y = tanh(W2 * z + b2);
            
            % Calculate error
            error = target - y;
            mean_sq_error = mean_sq_error + mean(error.^2);
            
            % Backpropagation of error
            delta_k = error .* (1 - y.^2);
            delta_j = (W2' * delta_k) .* (1 - z.^2);
            
            % Update weights and biases
            W2 = W2 + eta * (delta_k * z');
            b2 = b2 + eta * delta_k;
            W1 = W1 + eta * (delta_j * input');
            b1 = b1 + eta * delta_j;
        end
        
        % Calculate mean squared error for the epoch
        mean_sq_error = mean_sq_error / size(input_patterns, 1);
        
        % Display the epoch and the mean squared error
        fprintf('Epoch %d: Mean Squared Error = %f\n', epoch, mean_sq_error);
        
        % Check for convergence
        if mean_sq_error < epsilon
            fprintf('Convergence reached at epoch %d with MSE: %f\n', epoch, mean_sq_error);
            break;
        end
    end
end


