function [S] = obtenerSensitividades(Fm,w,e,M,m,Sg)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    if(M==m)
        S=-2*(Fm)*e;
    else
        S=Fm*(w{1,m+1}.')*(Sg{1,m+1});
    end
    
end

