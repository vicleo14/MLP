function [a] = logsig(W,p,b)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    a=(W*p)+b;
    [f,c]=size(a);
    for i=1:f
        a(i,1)=1/(1+exp(-(a(i,1))));
    end
end
