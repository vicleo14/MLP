function [] = P05_MLP()
%P5: Multilayer Perceptron
%Fecha de elaboraci�n: 2019/05/05
%Autor: Villegas Hernandez Carlos Uriel
%Autor: Morales Flores Victor Leonel
%Autor: Flores Garcia Alberto
%Asignatura: Neural Networks
%Escuela: ESCOM-IPN(MX)
    %Variable para almacenar el error de epoca
    epoca=0;
    Eepoch=0;
    archivoP=input('Ingrese el numero del polinomio deseado: ','s');
    %archivoT=input('Ingrese el nombre del archivo que contiene los valores deseados[targets](sin extension .txt): ','s');
    %rangoInf= input('Indica el valor m�nimo en el rango de la se�al: ');
    %rangoSup= input('Indica el valor m�nimo en el rango de la se�al: ');
    fprintf("\nLas opciones de separaci�n de datos son:");
    fprintf("\n 1)80%% Entrenamiento, 10%% Validaci�n, 10%% Prueba ");
    fprintf("\n 2)70%% Entrenamiento, 15%% Validaci�n, 15%% Prueba ");
    opcDatos= input('\nIndica la forma de separar los datos que deseas: ');
    archivoArq1=input('Ingrese el nombre del archivo que contiene la arquitectura[V1](sin extension .txt): ','s');
    archivoArq2=input('Ingrese el nombre del archivo que contiene la arquitectura[V2](sin extension .txt): ','s');
    alfa= input('Indica el factor de aprendizaje: ');
    eepoch_max= input('Indica el m�ximo de �pocas: ');
    eepoch= input('Indica el error m�ximo tolerable: ');
    eepoch_val= input('Indica cada cu�nto ser� la �poca de validaci�n: ');
    %num_val= input('Indica el n�mero m�ximo de intentos del error de validaci�n:');
    
    [p,targets]=lecturaDataSet(archivoP);
    [R,S,func] = lecturaVectores(archivoArq1,archivoArq2);
    %Visualizacion del polinomio a tratar
    GraficarPolinomio(p,targets,targets);
    
    %Generacion aleatoria de pesos y bias
    [fS,cS]=size(S);
    %Numero de capas
    M=fS;
    W=cell(1,M);
    b=cell(1,M);
    rAux=R;
    for i=1:M
        W{1,i}=generacionW(S(i,1),R);
        b{1,i}=generacionBias(S(i,1));
        R=S(i,1);
    end
    R=rAux;
    %Dividimos los datos
    [ptrain,ttrain,pval,tval,ptest,ttest]=separarDatos(p,targets,opcDatos);
    
    
    %Tamanio de p
    [fptrain,cptrain]=size(ptrain);
    [fpval,cpval]=size(pval);
    [fptest,cptest]=size(ptest);
    
    %Propagacion
    GuardarEepoch(epoca,Eepoch,"w",1);
    GuardarEepoch(epoca,Eepoch,"w",0);
    for epoca=1:eepoch_max
        fprintf(strcat("\nEpoca ",int2str(epoca),":"));
        Eepoch=0;
        
        if(mod(epoca, eepoch_val)==0)
            fprintf("\n***Validacion***");
             for valp=1:fpval
                a=propagacionAdelante(M,W,b,pval(valp,1),func);
                %M porque ahi se encuentra la salida de la ultima capa de
                %la red
                [e,he]=errorAprendizaje(tval(valp,1),a{1,M});
                Eepoch=Eepoch+e;
            end 
            Eepoch=abs(Eepoch/fpval);
            fprintf("\n>>>>>Error de epoca %d: %f",epoca,Eepoch);
            GuardarEepoch(epoca,Eepoch,"a",1);
            
        else
            %Entrenamiento
            fprintf("\n---Entrenamiento---");
            for valp=1:fptrain
                a=propagacionAdelante(M,W,b,ptrain(valp,1),func);
                %M porque ahi se encuentra la salida de la ultima capa de
                %la red
                [e,he]=errorAprendizaje(ttrain(valp,1),a{1,M});
                Eepoch=Eepoch+e;
                [W,b] =Backpropagation(W,b,a,M,func,e,alfa,ptrain(valp,1));
            end 
            Eepoch=abs(Eepoch/fptrain);
            
            GuardarEepoch(epoca,Eepoch,"a",0);
            fprintf("\n>>>>>Error de epoca %d: %f",epoca,Eepoch);
            if(Eepoch<eepoch )
                fprintf("\n>>>>>>El valor de error de la red es menor al error tolerable. Acaba entrenamiento");
                break;
            end
        end
    end
    GraficarEepoch(1);
    fprintf("\n---Prueba---");
    size(ptest)
    size(ttest)
    for valp=1:fptest
        a=propagacionAdelante(M,W,b,ptest(valp,1),func);
        %M porque ahi se encuentra la salida de la ultima capa de
        %la red
        [e,he]=errorAprendizaje(ttest(valp,1),a{1,M});
        fprintf("\n a: %f | target: %f | error: %f",a{1,M},ttest(valp,1),e);
    end 
end

