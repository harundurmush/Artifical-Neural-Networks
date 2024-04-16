clc;
clear all;
close all;
%%
x1 = -5:1:5;
x2 = -5:1:5;
d1 = @(x1,x2) x1;
d2 = @(x1,x2) x1 - x2 + 2;
d3 = @(x1,x2) -x2 -3;
d4 = @(x1,x2) -x1 -x2 -5;
d5 = @(x1,x2) -x1 -3;
d6 = @(x1,x2) -x1 +x2 -2;
d7 = @(x1,x2) x2;
d8 = @(x1,x2) x1 + x2 -1;
%%
o1 = sign(d1);
o2 = sign(d2);
o3 = sign(d3);
o4 = sign(d4);
o5 = sign(d5);
o6 = sign(d6);
o7 = sign(d7);
o8 = sign(d8);

