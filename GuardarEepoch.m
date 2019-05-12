function GuardarEepoch(epoca,Eepoch,opc,tipo)
%P3: Aprendizaje del perceptron 
%   Con el fin de un c�digo mas legible se crea esta funci�n encargar�
%   de escribir en un archivo de texto 'Eepoch.txt' con todos los valoers 
%   de Eepoch junto con la epoca. opc permite que decidamos entre 'w' para
%   reescribir el archivo en caso que exista o 'a' para escribir al final 
%   del mismo si es que existe.
%Se agrega tipo, donde 0 quiere decir que el error de epoca que se
%escribira es de entrenamiento y 1 quiere decir que es de validacion
%Fecha de elaboraci�n: 2019/04/22
%Autor: Morales Flores Victor Leonel
%Asignatura: Neural Networks
%Escuela: ESCOM-IPN(MX)
    nombreArchivo="";
    if(tipo==0)
        nombreArchivo="Eepoch_train.txt";
    elseif(tipo==1)
        nombreArchivo="Eepoch_val.txt";
    end
    results=fopen(nombreArchivo,opc);
    fprintf(results,"%d ",epoca);
    
    fprintf(results,"%d ",Eepoch);
    fprintf(results,"\n");
    fclose(results);
end
