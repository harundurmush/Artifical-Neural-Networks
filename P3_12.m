clc;
clear all;
close all;
%%
class1Patterns = [0.8 0.5 0 1;0.9 0.7 0.3 1;1 0.8 0.5 1];
class2Patterns = [0 0.2 0.3 1;0.2 0.1 1.3 1;0.2 0.7 0.8 1];
w=[0 0 0 0];

for i= 1:1000
    for c=1:3
        err1(c)=error(1,fnet(w,class1Patterns(c,:)));
    end

    for c=1:3
        err2(c)=error(-1,fnet(w,class2Patterns(c,:)));
    end
    if(all(err1<0.01)&&all(err2<0.01))
        break;
    end
    for c= 1:3
        w= wupdate(w,1,fnet(w,class1Patterns(c,:)),class1Patterns(c,:),5);
    end
    for c= 1:3
        w= wupdate(w,-1,fnet(w,class2Patterns(c,:)),class2Patterns(c,:),5);
    end      
end



function fout = fnet(w,y)
    fout=2/(1+exp(-(w*y.')))-1;
end

function errk = error(dk,ok)
    errk=(dk-ok)^2/2;
end

function wk1 = wupdate(wk,dk,ok,yk,n)
    wk1=wk+n*((dk-ok)*(1-ok^2)*yk)/2;
end