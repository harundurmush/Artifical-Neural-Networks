clc;
clear all;
close all;

%%
% Define the input patterns and their corresponding classes
% Note: You would need to extract the actual coordinates that define the
% regions for both classes from the figure provided.

% Sample input patterns for class 1 and class 2
% This is just an illustrative example. You need to define these based on the figure.
class1_patterns = [0, 2; 3, 2.5]; % replace with actual coordinates
class2_patterns = [1, 1; 2, 3];   % replace with actual coordinates


% Combine the input patterns for both classes
input_patterns = [class1_patterns; class2_patterns];






class1_pattern = [1,1 0,0,-1,-1, 0 ,0]; % replace with actual coordinates
class2_pattern = [0,0,1, 1,-1-1,1,-1]; 
bias = [0,3,0,3,2,-1,-2,-5];
% Define the targets for the classes
% Class 1 is labeled as 0 and Class 2 as 1
targets = [0; 0; 1; 1]; % Assuming two patterns for each class as an example

% Define the architecture of the network
num_inputs = 2;         % Two input features x1 and x2
num_hidden_neurons = 10; % Number of neurons in the hidden layer (adjustable)
num_outputs = 1;        % Single output for two classes

% Initialize the weights and biases for the hidden layer
hidden_weights = rand(num_hidden_neurons, num_inputs) - 0.5;
hidden_bias = rand(num_hidden_neurons, 1) - 0.5;

% Initialize the weights and biases for the output layer
output_weights = rand(num_outputs, num_hidden_neurons) - 0.5;
output_bias = rand(num_outputs, 1) - 0.5;

% Define the learning rate
learning_rate = 0.01;

% Define the number of epochs for training
num_epochs = 1000;

% Training the network using backpropagation
% Training the network using backpropagation
for epoch = 1:num_epochs
    for i = 1:size(input_patterns, 1)
        % Forward pass
        % Ensure the input pattern is a column vector for matrix multiplication
        input_pattern = input_patterns(i, :)';
        
        % Matrix multiplication for weights and addition for bias
        hidden_layer_input = hidden_weights * input_pattern + hidden_bias;
        
        % Sigmoid activation function
        hidden_layer_output = 1 ./ (1 + exp(-hidden_layer_input));
        
        % Matrix multiplication for weights and addition for bias
        output_layer_input = output_weights * hidden_layer_output + output_bias;
        
        % Sigmoid activation function for output
        output = 1 ./ (1 + exp(-output_layer_input));

        % Calculate the error
        error = targets(i) - output;

        % Backward pass (backpropagation)
        % Gradient for the output layer
        output_delta = error .* output .* (1 - output);

        % Gradient for the hidden layer
        hidden_delta = (output_weights' * output_delta) .* hidden_layer_output .* (1 - hidden_layer_output);

        % Update the weights and biases for output layer
        output_weights = output_weights + learning_rate * output_delta * hidden_layer_output';
        output_bias = output_bias + learning_rate * output_delta;

        % Update the weights and biases for hidden layer
        hidden_weights = hidden_weights + learning_rate * hidden_delta * input_pattern';
        hidden_bias = hidden_bias + learning_rate * hidden_delta;
    end
end



% Display the final weights
fprintf('Training completed.\n');
disp('Final hidden weights:');
disp(hidden_weights);
disp('Final output weights:');
disp(output_weights);


% ... (previous code for training the network) ...

% Define a grid over the input space
x1_range = linspace(min(input_patterns(:, 1)), max(input_patterns(:, 1)), 100);
x2_range = linspace(min(input_patterns(:, 2)), max(input_patterns(:, 2)), 100);
[X1, X2] = meshgrid(x1_range, x2_range);

% Calculate the output of the network over the entire grid
Z = zeros(size(X1));
for i = 1:size(X1, 1)
    for j = 1:size(X1, 2)
        % Forward pass for each point on the grid
        input_pattern = [X1(i, j); X2(i, j)];
        hidden_layer_input = hidden_weights * input_pattern + hidden_bias;
        hidden_layer_output = 1 ./ (1 + exp(-hidden_layer_input));
        output_layer_input = output_weights * hidden_layer_output + output_bias;
        Z(i, j) = 1 ./ (1 + exp(-output_layer_input));
    end
end

% Plot the decision boundary


new_input_pattern = [1; 2];  % Column vector representing a new point
%new_input_pattern = [0; 0];
%new_input_pattern = [1.5; 2.5];
%new_input_pattern = [1; 2];
% Calculate the output of the network for the new input pattern
hidden_layer_input = hidden_weights * new_input_pattern + hidden_bias;
hidden_layer_output = 1 ./ (1 + exp(-hidden_layer_input));
output_layer_input = output_weights * hidden_layer_output + output_bias;
network_output = 1 ./ (1 + exp(-output_layer_input));

% Display the network output
fprintf('The network output for the given input pattern is:\n');
disp(network_output);

% Determine the class based on the network output
% Assuming a threshold of 0.5 to decide between Class 1 and Class 2
class_threshold = 0.5;
predicted_class = network_output > class_threshold; % Class 2 if true, Class 1 if false

% Display the predicted class
fprintf('The predicted class for the given input pattern is:\n');
if predicted_class
    disp('Class 2');
else
    disp('Class 1');
end

% You can now use the trained network to classify new data points
% or visualize the decision boundary.
