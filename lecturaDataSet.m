function [p,targets] = lecturaDataSet(nomArc)
%P3: Aprendizaje del perceptron 
%   Funcion para la lectura del documento que contiene el dataet y que
%   devielve las entradas a la red, los targets y el numero de neuronas de
%   la red.
%Fecha de elaboraci�n: 2019/04/06
%Autor: Morales Flores Victor Leonel
%Asignatura: Neural Networks
%Escuela: ESCOM-IPN(MX)
    path="entrenamiento/0";
    entrenamiento="_Polinomio_Entrada";
    targetsc="_Polinomio_Target";
    arcP = fopen(strcat(path,nomArc,entrenamiento,'.txt'),'r');
    p=fscanf(arcP,"%f",Inf);
    fclose(arcP);
    
     arcT = fopen(strcat(path,nomArc,targetsc,'.txt'),'r');
     targets=fscanf(arcT,"%f",Inf);
     fclose(arcT);
    
end