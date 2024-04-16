clc;clear all;close all;

% Pattern for 'A'
A_pattern = [
    1; 1; 1;  1;  % Top row
    1; -1; -1;  1;  % Second row
    1;  1;  1;  1;  % Third row (line across)
   1;  -1; -1; 1;  % Bottom row
];

% Pattern for 'I'
I_pattern = [
   -1;  1; -1; -1;  % Top row
   -1;  1; -1; -1;  % Second row
   -1;  1; -1; -1;  % Third row
   -1;  1; -1; -1;  % Bottom row
];

% Pattern for 'O'
O_pattern = [
    1;  1;  1;  1;  % Top row
    1; -1; -1;  1;  % Second row
    1; -1; -1;  1;  % Third row
    1;  1;  1;  1;  % Bottom row
];

% Combine into a matrix where each column is an input pattern
input_patterns_a = [A_pattern, I_pattern, O_pattern];
desired_responses = [
    1 -1 -1;  % Target for A
    -1 1 -1;  % Target for I
    -1 -1 1;  % Target for O
];
% Architecture for guideline (a)
I_a = 16;
J_a = 9;
K_a = 3;

% Learning rate
eta = 0.1;

% Call the function for architecture (a)
[V_a, W_a, Outputs_a] = ebpta(I_a, J_a, K_a, eta, input_patterns_a, desired_responses);


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
    disp(Outputs);
end


% function [V, W] = ebpta(I, J, K, eta, input_patterns, desired_responses)
%     % Define the architecture
%     V = 2*rand(J, I) - 1; % Weights from input to hidden layer
%     W = 2*rand(K, J) - 1; % Weights from hidden to output layer
% 
%     % Training parameters
%     max_epochs = 10000; % Maximum number of epochs
%     min_error = 1e-6; % Minimum error to stop training
% 
%     % Begin training
%     for epoch = 1:max_epochs
%         total_error = 0;
% 
%         for p = 1:size(input_patterns, 2)
%             % Forward pass
%             Z = input_patterns(:, p);
%             Y = tanh(V * Z); % Bipolar sigmoid activation
%             O = tanh(W * Y);
% 
%             % Error calculation
%             D = desired_responses(:, p);
%             error = D - O;
%             total_error = total_error + sum(error .^ 2);
% 
%             % Backward pass
%             delta_o = error .* (1 - O .^ 2); % Gradient for output layer
%             delta_y = (1 - Y .^ 2) .* (W' * delta_o); % Gradient for hidden layer
% 
%             % Weights update
%             W = W + eta * (delta_o * Y');
%             V = V + eta * (delta_y * Z');
%         end
% 
%         % Check for convergence
%         if total_error < min_error
%             fprintf('Convergence reached after %d epochs.\n', epoch);
%             break;
%         end
%     end
% 
%     if epoch == max_epochs
%         fprintf('Max epochs reached. Training stopped.\n');
%     end
% 
%     % Optionally, you can print the final weights
%     disp('Final weights from input to hidden layer (V):');
%     disp(V);
%     disp('Final weights from hidden to output layer (W):');
%     disp(W);
% end
% 


% function ebpta(I, J, K, eta, input_patterns, desired_responses)
%     % Define the architecture
%     V = 2*rand(J, I) - 1; % Weights from input to hidden layer
%     W = 2*rand(K, J) - 1; % Weights from hidden to output layer
% 
%     % Training parameters
%     max_epochs = 100; % Maximum number of epochs
%     min_error = 1e-6; % Minimum error to stop training
% 
%     % Begin training
%     for epoch = 1:max_epochs
%         total_error = 0;
% 
%         for p = 1:size(input_patterns, 2)
%             % Forward pass
%             Z = input_patterns(:, p);
%             Y = tanh(V * Z); % Bipolar sigmoid activation
%             O = tanh(W * Y);
% 
%             % Error calculation
%             D = desired_responses(:, p);
%             error = D - O;
%             total_error = total_error + sum(error .^ 2);
% 
%             % Backward pass
%             delta_o = error .* (1 - O .^ 2); % Gradient for output layer
%             delta_y = (1 - Y .^ 2) .* (W' * delta_o); % Gradient for hidden layer
% 
%             % Weights update
%             W = W + eta * (delta_o * Y');
%             V = V + eta * (delta_y * Z');
%         end
% 
%         % Check for convergence
%         if total_error < min_error
%             fprintf('Convergence reached after %d epochs.\n', epoch);
%             break;
%         end
%     end
% 
%     if epoch == max_epochs
%         fprintf('Max epochs reached. Training stopped.\n');
%     end
% end
% 
