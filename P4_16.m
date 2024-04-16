clc;
clear all;
close all;
% Define the input patterns (flattened 4x4 binary grids for A, I, O)
% These patterns are placeholders and should be replaced with actual binary representations of the characters
% Example binary patterns for the characters A, I, O as 4x4 grids
% These will be bipolar with 1 representing black and -1 representing white
input_patterns = [
    % A: Assuming it has a pattern with a filled top row and a vertical line in the middle
    reshape([-1 1 1 1; -1 -1 1 -1; -1 1 1 1; -1 -1 1 -1]', 1, 16);
    % I: Assuming it is just a vertical line in the middle
    reshape([-1 -1 1 -1; -1 -1 1 -1; -1 -1 1 -1; -1 -1 1 -1]', 1, 16);
    % O: Assuming it is a square
    reshape([-1 1 1 -1; 1 -1 -1 1; 1 -1 -1 1; -1 1 1 -1]', 1, 16);
];

% Define the target outputs for each character (bipolar representation)
targets = [
    1 -1 -1;  % Target for A
    -1 1 -1;  % Target for I
    -1 -1 1   % Target for O
];

% Define the architecture of the neural network
I = 16; % Number of input neurons (16 pixels)
J = 9;  % Number of hidden neurons (can be adjusted)
K = 3;  % Number of output neurons (one for each character)

% Define the learning rate (eta) and the acceptable error threshold (epsilon)
eta = 0.1;
epsilon = 0.01;

% Train the network
[W1, b1, W2, b2] = train_network(input_patterns, targets, I, J, K, eta, epsilon);
%% buraya gir
new_input_for_A = reshape([-1 1 1 1; -1 -1 1 -1; -1 1 1 1; -1 -1 1 -1]', 1, 16);
test_network(W1, b1, W2, b2, new_input_for_A);
% Function to train the network
function test_network(W1, b1, W2, b2, new_input)
    z = tanh(W1 * new_input' + b1);
    y = softmax(W2 * z + b2);
    fprintf('Network output for the given input pattern:\n');
    disp(y);
    if y(1,1) > y(2,1) && y(1,1) > y(3,1)
    disp("Class 1");
    elseif y(2,1) > y(1,1) && y(2,1) > y(3,1)
    disp("Class 2");
    else
        disp("Class 3");
end

end
function [W1, b1, W2, b2] = train_network(input_patterns, targets, I, J, K, eta, epsilon)
    % Initialize weights and biases
    W1 = rand(J, I) * 2 - 1;
    b1 = rand(J, 1) * 2 - 1;
    W2 = rand(K, J) * 2 - 1;
    b2 = rand(K, 1) * 2 - 1;

    % Training loop
    epoch = 0;
    mean_sq_error = inf;
    while mean_sq_error > epsilon
        mean_sq_error = 0;
        for p = 1:size(input_patterns, 1)
            % Forward pass
            input = input_patterns(p, :)';
            target = targets(p, :)';
            z = tanh(W1 * input + b1);
            y = tanh(W2 * z + b2);

            % Error calculation
            error = target - y;
            mean_sq_error = mean_sq_error + mean(error .^ 2);

            % Backward pass (delta rule)
            delta_k = error .* (1 - y .^ 2);
            delta_j = (W2' * delta_k) .* (1 - z .^ 2);

            % Weights update
            W2 = W2 + eta * (delta_k * z');
            b2 = b2 + eta * delta_k;
            W1 = W1 + eta * (delta_j * input');
            b1 = b1 + eta * delta_j;
        end
        mean_sq_error = mean_sq_error / size(input_patterns, 1);
        epoch = epoch + 1;
        fprintf('Epoch %d: Mean Squared Error = %f\n', epoch, mean_sq_error);
    end
    fprintf('Training completed in %d epochs with MSE: %f\n', epoch, mean_sq_error);
end

% Function to test the network with a new input


% Example: Test the trained network with a new input for character A
% Replace the 'new_input_for_A' with the actual input for A

