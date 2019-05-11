function GraficarPolinomio (p,target,results)
    %Funcion que permite graficar toddos los valores obtenidos. 
    %Recibe S para poder leer de forma adecuada el archivo que contiene los datos.
%Fecha de elaboraci�n: 2019/04/22
%Autor: Morales Flores Victor Leonel
%Asignatura: Neural Networks
%Escuela: ESCOM-IPN(MX)
    etiquetas=["targets","results"];
    figure(1);
    plot(p,target,"o-"); 
    hold on;
    plot(p,results,"x"); 
    %Configuracion de la grafica a mostrar
    xticks(min(p):0.5:max(p));
    yticks(min(target):max(target));
    title('Polinoio');
    xlabel('Rango de se�al');
    ylabel('Valores');
    legend(etiquetas);
end
