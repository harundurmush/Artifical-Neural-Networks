clc;
clear all;
close all;
y_1 = [1 0 0 1];
y_2 = [1 1 0 1];
y_3 = [1 0 1 1];
y_4 = [1 1 1 1]; 
y_5 = [0 1 0 1];
y_6 = [0 1 1 1];
y_7 = [0 0 0 1];
y_8 = [0 0 1 1];
Y = [y_1' y_2' y_3' y_4' y_5' y_6' y_7' y_8']; 
label_matrix = [1 2 3 4];
temp_lab = [];
for i=1:length(label_matrix)
    lab = [0 0 0 0];
    lab(i) = 1; 
    temp_lab = [temp_lab lab' lab'];

end
input_size = 4;
hidden_size = 4;
output_size = 4;
learning_rate = 1;
num_samples = 8;
w_1 = randn(hidden_size, input_size);
w_2 = randn(output_size, hidden_size);
iter_size = 1000; 
for iter = 1:iter_size
    for i = 1:num_samples
        Y_t = Y(:, i);
        label = temp_lab(:, i);

        z_1 = w_1 * Y_t ;
        a_1 = sigmoid(z_1);
        z_2 = w_2 * a_1 ;
        a_2 = sigmoid(z_2);

       
        delta2 = a_2 - label;
        delta1 = (w_2' * delta2).*(a_1.*(1 - a_1));

        w_2 = w_2 - learning_rate * delta2 * a_1';
        w_1 = w_1 - learning_rate * delta1 * Y_t';
    end
end
solution = []; 
for i = 1:num_samples
        Y_t = Y(:, i);
        z_1 = w_1 * Y_t ;
        a_1 = sigmoid(z_1);
        z_2 = w_2 * a_1 ;
        a_2 = sigmoid(z_2);

        solution = [solution a_2==max(a_2)];

end
disp(solution);
function out = sigmoid(input)
    out = 1./(1 + exp(-input));
end