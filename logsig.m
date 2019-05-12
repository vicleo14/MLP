function [a] = logsig(W,p,b)
%P5: Multilayer perceptron
%   Implementacio de la funcion logsig
%Fecha de elaboración: 2019/05/11
%Autor: Morales Flores Victor Leonel
%Asignatura: Neural Networks
%Escuela: ESCOM-IPN(MX)
    a=(W*p)+b;
    [f,c]=size(a);
    for i=1:f
        a(i,1)=1/(1+exp(-(a(i,1))));
    end
end
