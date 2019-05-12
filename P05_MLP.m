function [] = P05_MLP()
%P5: Multilayer Perceptron
%Fecha de elaboraci�n: 2019/05/05
%Autor: Villegas Hernandez Carlos Uriel
%Autor: Morales Flores Victor Leonel
%Autor: Flores Garcia Alberto
%Asignatura: Neural Networks
%Escuela: ESCOM-IPN(MX)
    
    archivoP=input('Ingrese el numero del polinomio deseado: ','s');
    %archivoT=input('Ingrese el nombre del archivo que contiene los valores deseados[targets](sin extension .txt): ','s');
    %rangoInf= input('Indica el valor m�nimo en el rango de la se�al: ');
    %rangoSup= input('Indica el valor m�nimo en el rango de la se�al: ');
    %fprintf("\nLas opciones de separaci�n de datos son:");
    %fprintf("\n 1)80% Entrenamiento, 10% Validaci�n, %10% Prueba ");
    %fprintf("\n 2)70% Entrenamiento, 15% Validaci�n, %15% Prueba ");
    %opcDatos= input('Indica la forma de separar los datos que deseas: ');
    archivoArq1=input('Ingrese el nombre del archivo que contiene la arquitectura[V1](sin extension .txt): ','s');
    archivoArq2=input('Ingrese el nombre del archivo que contiene la arquitectura[V2](sin extension .txt): ','s');
    alfa= input('Indica el factor de aprendizaje: ');
    eepoch_max= input('Indica el m�ximo de �pocas: ');
    %e_epoch= input('Indica el error m�ximo tolerable: ');
    %eepoch_val= input('Indica cada cu�nto ser� la �poca de validaci�n: ');
    %num_val= input('Indica el n�mero m�ximo de intentos del error de validaci�n:');
    
    [p,targets]=lecturaDataSet(archivoP);
    [R,S,func] = lecturaVectores(archivoArq1,archivoArq2);
    %Visualizacion del polinomio a tratar
    %GraficarPolinomio(p,targets,targets);
    
    %Generacion aleatoria de pesos y bias
    [fS,cS]=size(S);
    %Numero de capas
    M=fS;
    W=cell(1,fS);
    b=cell(1,fS);
    rAux=R;
    for i=1:M
        W{1,i}=generacionW(S(i,1),R);
        b{1,i}=generacionBias(S(i,1));
        R=S(i,1);
    end
    R=rAux;
    %W
    %b
    
    %Propagacion
    %Tamanio de p
    [fp,cp]=size(p);
    for epoca=1:eepoch_max
        fprintf(strcat("Epoca ",int2str(epoca),":"));
        %Entrenamiento 
        for valp=1:fp
            a=propagacionAdelante(M,W,b,p(valp,1),func);
            %M porque ahi se encuentra la salida de la red
            [e,he]=errorAprendizaje(targets(valp,1),a{1,M});
            %[W,b] = 
            Backpropagation(W,b,a,M,func,e,alfa);
            
        end
    end
    
end

