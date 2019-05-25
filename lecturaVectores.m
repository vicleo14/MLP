function [r,S,func] = lecturaVectores(archV1,archV2)
%P5: Multilayer perceptron
%   Funcion para leer los vectores que V1 y V2 que definen la arquitectura
%   de la red
%Fecha de elaboración: 2019/05/11
%Autor: Morales Flores Victor Leonel
%Asignatura: Neural Networks
%Escuela: ESCOM-IPN(MX)xt'),'r')
    arcV1 = fopen(strcat(archV1,'.txt'),'r');
    v1=fscanf(arcV1,"%d",Inf)
    fclose(arcV1);
    
    r=v1(1)
    S=v1(2:end)
    arcV2 = fopen(strcat(archV2,'.txt'),'r');
    func=fscanf(arcV2,"%d",Inf)
    fclose(arcV2);
end

