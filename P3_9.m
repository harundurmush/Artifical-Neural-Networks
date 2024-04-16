clc;
clear all; 
close all;
%%
x_1 = [0.8 0.5 0 1; 0.9 0.7 0.3 1; 1 0.8 0.5 1; 0 0.2 0.3 1; 0.2 0.1 1.3 1; 0.2 0.7 0.8 1];
x_1 = [x_1;x_1;x_1];
weight = zeros(100,4);
for a =1:6:17
    for i=a:a+2
        if weight(i,:) * transpose(x_1(i,:)) == 0
            weight(i+1,:) = weight(i,:) + x_1(i,:);
        
        elseif sign(weight(i,:) * transpose(x_1(i,:))) == 1
            weight(i+1,:) = weight(i,:);
        else
            weight(i+1,:) = weight(i,:) + (1/2 * ( 1 - sign(weight(i,:)* transpose(x_1(i,:)))) * x_1(i,:));
        end
    end
    
    for i=a+3:a+5
            weight(i+1,:) = weight(i,:) +( 1/2 * ( -1 - sign(weight(i,:)* transpose(x_1(i,:)))) * x_1(i,:));
    
    end
end
%%
display=sign(weight(19,:)*transpose(x_1(6,:)));

for i=1:6
    x = [0.8 0.5 0 1; 0.9 0.7 0.3 1; 1 0.8 0.5 1; 0 0.2 0.3 1; 0.2 0.1 1.3 1; 0.2 0.7 0.8 1];
    b(i) = sign(weight(19,:) * transpose(x(i,:)));
end
