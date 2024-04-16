clc;
clear all;
%%
p_1 = [1,0,0,1,0,0,1,1,1];
p_2 = [0,1,0,0,1,0,0,1,0];
p_3 = [1,1,1,0,0,0,0,1,1];
p_4 = [0,0,1,0,0,1,1,1,1];
p_5 = [1,1,1,0,0,1,0,1,0];
p_6 = [0,0,1,0,0,1,1,1,0];
input= [1,1,1,1,1,1,1,1,1];
weight = zeros(60,9);
%%
for i=1:6:60
    if sign(weight(i,:) * transpose(p_1(1,:))) == 0
        weight(i+1,:) = weight(i,:) + p_1(1,:);
    elseif sign(weight(i,:) * transpose(p_1(1,:))) < 0
        weight(i+1,:) = weight(i,:) + p_1(1,:);
    else 
        weight(i+1,:) = weight(i,:);
    end
%%
    if sign(weight(i+1,:) * transpose(p_2(1,:))) == 0
        weight(i+2,:) = weight(i+1,:) + p_2(1,:);
    elseif sign(weight(i+1,:) * transpose(p_2(1,:))) < 0
        weight(i+2,:) = weight(i+1,:) + p_2(1,:);
    else 
        weight(i+2,:) = weight(i+1,:);
    end
%%
    if sign(weight(i+2,:) * transpose(p_3(1,:))) < 0
        weight(i+3,:) = weight(i+2,:);
    
    else 
        weight(i+3,:) = weight(i+2,:) - p_3(1,:);
    end
%%
    if sign(weight(i+3,:) * transpose(p_4(1,:))) < 0
        weight(i+4,:) = weight(i+3,:); 
    
    else 
        weight(i+4,:) = weight(i+3,:) - p_4(1,:);
    end
%%
    if sign(weight(i+4,:) * transpose(p_5(1,:))) < 0
        weight(i+5,:) = weight(i+4,:); 
    
    else 
        weight(i+5,:) = weight(i+4,:) - p_5(1,:);
    end
%%
    if sign(weight(i+5,:) * transpose(p_6(1,:))) < 0
        weight(i+6,:) = weight(i+5,:); 
    
    else 
        weight(i+6,:) = weight(i+5,:) - p_6(1,:);
    end
end