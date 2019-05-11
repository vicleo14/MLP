function [r,S,func] = lecturaVectores(archV1,archV2)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
    arcV1 = fopen(strcat(archV1,'.txt'),'r');
    v1=fscanf(arcV1,"%d",Inf)
    fclose(arcV1);
    
    r=v1(1)
    S=v1(2:end)
    arcV2 = fopen(strcat(archV2,'.txt'),'r');
    func=fscanf(arcV2,"%d",Inf)
    fclose(arcV2);
end

