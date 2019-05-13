function [a] = tansig(W,p,b)
%P5: Multilayer perceptron
%   Implementacion de la funcion tansig
%Fecha de elaboración: 2019/05/11
%Autor: Morales Flores Victor Leonel
%Asignatura: Neural Networks
%Escuela: ESCOM-IPN(MX)
    a=(W*p)+b;
    [f,c]=size(a);
    for i=1:f
        n=a(i,1);
        a(i,1)=(exp(n)-exp(-n))/(exp(n)+exp(-n));
    end
end
