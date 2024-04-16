clc;clear all; close all;
%%

x1_1 = [1 0 0 1;1 1 0 1; 1 0 1 1; 1 1 1 1; 0 1 0 1; 0 1 1 1; 0 0 0 1; 0 0 1 1];
x1 = repmat(x1_1, 18, 1);
x2_1 = [1 0 1 1; 1 1 1 1; 0 1 0 1; 0 1 1 1; 0 0 0 1; 0 0 1 1;1 0 0 1;1 1 0 1];
x2 = repmat(x2_1, 18, 1);
x3_1 = [0 1 0 1; 0 1 1 1;1 0 0 1;1 1 0 1; 1 0 1 1; 1 1 1 1; 0 0 0 1; 0 0 1 1];
x3 = repmat(x3_1, 18, 1);
x4_1= [0 0 0 1; 0 0 1 1;1 0 0 1;1 1 0 1; 1 0 1 1; 1 1 1 1; 0 1 0 1; 0 1 1 1];
x4 = repmat(x4_1, 18, 1);
w1 = zeros(1000,4);
w2 = zeros(1000,4);
w3 = zeros(1000,4);
w4 = zeros(1000,4);
%% weight 1
for a =1:8:138
    for i=a:a+1
        if sign(w1(i,:) * transpose(x1(i,:))) ==1
            w1(i+1,:) = w1(i,:);
        else
            w1(i+1,:) = w1(i,:) + (x1(i,:));
        end
    
    end
    
    for i=a+2:a+7
        if sign(w1(i,:) * transpose(x1(i,:))) < 0
            w1(i+1,:) = w1(i,:);
        
        else
            w1(i+1,:) = w1(i,:) - (x1(i,:));
        end
    
    end
    a = a + 8;
end
%% weight 2
for a =1:8:138
    for i=a:a+1
        if sign(w2(i,:) * transpose(x2(i,:))) ==1
            w2(i+1,:) = w2(i,:);
        else
            w2(i+1,:) = w2(i,:) + (x2(i,:));
        end
    
    end   
    for i=a+2:a+7
        if sign(w2(i,:) * transpose(x2(i,:))) < 0
            w2(i+1,:) = w2(i,:);
        
        else
            w2(i+1,:) = w2(i,:) - (x2(i,:));
        end
    
    end
    a = a + 8;
end
%% weight 3
for a =1:8:138
    for i=a:a+1
        if sign(w3(i,:) * transpose(x3(i,:))) ==1
            w3(i+1,:) = w3(i,:);
        else
            w3(i+1,:) = w3(i,:) + (x3(i,:));
        end
    
    end   
    for i=a+2:a+7
        if sign(w3(i,:) * transpose(x3(i,:))) < 0
            w3(i+1,:) = w3(i,:);
        
        else
            w3(i+1,:) = w3(i,:) - (x3(i,:));
        end
    
    end
    a = a + 8;
end
%% weight 4
for a =1:8:138
    for i=a:a+1
        if sign(w4(i,:) * transpose(x4(i,:))) ==1
            w4(i+1,:) = w4(i,:);
        else
            w4(i+1,:) = w4(i,:) + (x4(i,:));
        end
    
    end
    for i=a+2:a+7
        if sign(w4(i,:) * transpose(x4(i,:))) < 0
            w4(i+1,:) = w4(i,:);
        
        else
            w4(i+1,:) = w4(i,:) - (x4(i,:));
        end
    
    end
    a = a + 8;
end
w = [transpose(w1(145,:)) transpose(w2(145,:)) transpose(w3(145,:)) transpose(w4(145,:))];
%% results
fprintf("Final weights as column vectors:\n");
disp(w);
