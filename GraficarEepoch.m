function GraficarEepoch (s)
    %Funcion que permite graficar la evolucion del error en cada epoca de
    %aprrendizaje
%Fecha de elaboraci�n: 2019/04/22
%Autor: Morales Flores Victor Leonel
%Asignatura: Neural Networks
%Escuela: ESCOM-IPN(MX)
    etiquetas="";
     %Abrimos el archivo archivo results todos los valores que se crearon en
    %las n iteraciones 
    results = fopen('Eepoch_train.txt','r');
    %Formato de lectura del archivo
    filas=s+1;
    sizeRes=[filas Inf];
    %Se recupera en Res la matriz de resultados y se cierra el archivo
    %results.txt
    Res=fscanf(results,"%f",sizeRes);
    fclose(results);
    Res=Res.';
    
    %En t se almacena la primer fila del archivo que corresponde a la
    %iteracion en la que se generaron los resultados de esa columna
    epoca=Res(:,1);
    Eepoch=Res(:,2:end);
    
    figure(2)
    %Grafica de Eepoch
    plot(epoca,Eepoch(:,1),"-");
    etiquetas(1)="Epoch entrenamiento";
    hold on
    
    
    
    results = fopen('Eepoch_val.txt','r');
    %Se recupera en Res la matriz de resultados y se cierra el archivo
    %results.txt
    Res2=fscanf(results,"%f",sizeRes);
    fclose(results);
    Res2=Res2.';
    
    %En t se almacena la primer fila del archivo que corresponde a la
    %iteracion en la que se generaron los resultados de esa columna
    epoca_val=Res2(:,1);
    Eepoch_val=Res2(:,2:end);
    %Grafica de Eepoch val
    plot(epoca_val,Eepoch_val(:,1),"o");
    etiquetas(2)="Epoch validacion";
    hold on
    
    %Configuracion de la grafica a mostrar
    
    xticks(min(epoca):1:max(epoca));
    yticks(min(Eepoch):max(Eepoch));
    title('Evoluci�n de Eepoch');
    xlabel('�pocas');
    ylabel('Eepoch');
    legend(etiquetas);
end

