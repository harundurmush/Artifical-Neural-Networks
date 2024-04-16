% Define the input patterns for each class
class1 = [0.8 0.5 0; 0.9 0.7 0.3; 1 0.8 0.5]';
class2 = [0 0.2 0.3; 0.2 0.1 1.3; 0.2 0.7 0.8]';

% Augment the patterns with a bias term
bias = 1;
class1 = [bias * ones(1, size(class1, 2)); class1];
class2 = [bias * ones(1, size(class2, 2)); class2];

% Initialize the weight vector with zeros
weights = zeros(1, size(class1, 1));

% Define the learning rate
c = 1;

% Perceptron training algorithm
iteration = 0;
correctly_classified = false;
while ~correctly_classified
    correctly_classified = true;
    
    % Check each pattern from class 1
    for i = 1:size(class1, 2)
        if (weights * class1(:, i)) <= 0
            weights = weights + c * class1(:, i)';
            correctly_classified = false;
        end
    end
    
    % Check each pattern from class 2
    for i = 1:size(class2, 2)
        if (weights * class2(:, i)) >= 0
            weights = weights - c * class2(:, i)';
            correctly_classified = false;
        end
    end
    
    iteration = iteration + 1;
end

% Display the final weight vector
fprintf('Final weight vector after %d iterations:\n', iteration);
disp(weights);
