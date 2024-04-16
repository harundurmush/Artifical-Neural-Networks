clc; clear; close all;

% Define the input patterns for the two-dimensional points.
% Each column is one input pattern corresponding to the points in the figure.
input_patterns = [
    0.25 0.25 0.25 0.5 0.5 0.5 0.75 0.75 0.75;
    0.25 0.5 0.75 0.125 0.5 0.75 0.125 0.25 0.5;
    -1 -1 -1 -1 -1 -1 -1 -1 -1 
];

% Define the desired responses for the three classes
% The target vectors are for class 1, class 2, and class 3 respectively
desired_responses = [
    1 -1 -1;
    -1 -1 1;
    -1 1 -1;
    -1 1 -1;
    -1 -1 1;
    1 -1 -1;
    1 -1 -1;
    -1 1 -1;
    -1 -1 1
];
desired_responses=desired_responses';


% Define the number of neurons in the input, hidden, and output layers
I = 3; % Number of input neurons
J = 5; % Suitable number of neurons in the hidden layer can be chosen experimentally
K = 3; % Number of output neurons

% Learning rate
eta = 0.1;

% Call the ebpta function
[V, W, Outputs] = ebpta(I, J, K, eta, input_patterns, desired_responses);



function [V, W, Outputs] = ebpta(I, J, K, eta, input_patterns, desired_responses)
    % Define the architecture
    V = 2*rand(J, I) - 1; % Weights from input to hidden layer
    W = 2*rand(K, J) - 1; % Weights from hidden to output layer
    
    % Training parameters
    max_epochs = 10000; % Maximum number of epochs
    min_error = 1e-6; % Minimum error to stop training
    
    % Storage for outputs
    Outputs = zeros(K, size(input_patterns, 2));
    
    % Begin training
    for epoch = 1:max_epochs
        total_error = 0;
        
        for p = 1:size(input_patterns, 2)
            % Forward pass
            Z = input_patterns(:, p);
            Y = tanh(V * Z); % Bipolar sigmoid activation
            O = tanh(W * Y);
            Outputs(:, p) = O; % Store the output for this pattern
            
            % Error calculation
            D = desired_responses(:, p);
            error = D - O;
            total_error = total_error + sum(error .^ 2);
            
            % Backward pass
            delta_o = error .* (1 - O .^ 2); % Gradient for output layer
            delta_y = (1 - Y .^ 2) .* (W' * delta_o); % Gradient for hidden layer
            
            % Weights update
            W = W + eta * (delta_o * Y');
            V = V + eta * (delta_y * Z');
        end
        
        % Check for convergence
        if total_error < min_error
            fprintf('Convergence reached after %d epochs.\n', epoch);
            break;
        end
    end
    
    if epoch == max_epochs
        fprintf('Max epochs reached. Training stopped.\n');
    end
    Outputs = (Outputs + 1) / 2; 
    % Output the final weights and outputs
    disp('Final weights from input to hidden layer (V):');
    disp(V);
    disp('Final weights from hidden to output layer (W):');
    disp(W);
    disp('Outputs for each input pattern:');
    disp(Outputs');
end