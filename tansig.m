function [a] = tansig(W,p,b)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    a=(W*p)+b;
    [f,c]=size(a);
    for i=1:f
        n=a(i,1);
        a(i,1)=(exp(n)-exp(-n))/(exp(n)+exp(-n));
    end
end
