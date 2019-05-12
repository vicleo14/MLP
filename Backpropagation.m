function [Wbp,bbp] = Backpropagation(W,b,a,M,func,e,alfa)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    %Obtenemos Fn para todas las capas
    Fns=cell(1,M);
    for i=1:M
        Fns{1,i}=Fn(a{1,i},func(i,1));
    end
    %Obtenemos sensitividades para todas las capas
    S=cell(1,M);
    for m=M:-1:1
        %Fns{1,m}=Fn(a{1,m},func(i,m));
        S{1,m}=obtenerSensitividades(Fns{1,m},W,e,M,m,S);
    end
    Wbp=0;
    bbp=0;
end

