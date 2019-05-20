%P5: Multilayer Perceptron
%Fecha de elaboración: 2019/05/05
%Autor: Villegas Hernandez Carlos Uriel
%Autor: Morales Flores Victor Leonel
%Autor: Flores Garcia Alberto
%Asignatura: Neural Networks
%Escuela: ESCOM-IPN(MX)
    %Variable para almacenar el error de epoca
    epoca=0;
    Eepoch=0;
    archivoP=input('Ingrese el numero del polinomio deseado: ','s');
    %rangoInf= input('Indica el valor mínimo en el rango de la señal: ');
    %rangoSup= input('Indica el valor mínimo en el rango de la señal: ');
    fprintf("\nLas opciones de separación de datos son:");
    fprintf("\n 1)80%% Entrenamiento, 10%% Validación, 10%% Prueba ");
    fprintf("\n 2)70%% Entrenamiento, 15%% Validación, 15%% Prueba ");
    opcDatos= input('\nIndica la forma de separar los datos que deseas: ');
    archivoArq1=input('Ingrese el nombre del archivo que contiene la arquitectura[V1](sin extension .txt): ','s');
    archivoArq2=input('Ingrese el nombre del archivo que contiene la arquitectura[V2](sin extension .txt): ','s');
    alfa= input('Indica el factor de aprendizaje: ');
    eepoch_max= input('Indica el máximo de épocas: ');
    eepoch= input('Indica el error máximo tolerable: ');
    eepoch_val= input('Indica cada cuánto será la época de validación: ');
    %num_val= input('Indica el número máximo de intentos del error de validación:');
    archi=input('¿Usará archivo para pesos y bias? 1-Si. 0-No: ');
    [p,targets]=lecturaDataSet(archivoP);
    [R,S,func] = lecturaVectores(archivoArq1,archivoArq2);
    %Visualizacion del polinomio a tratar
    GraficarPolinomio(p,targets,targets);
    
    %Generacion aleatoria de pesos y bias----------------------------------
    [fS,cS]=size(S);
    %Numero de capas
    M=fS;
    if archi==0
        W=cell(1,M);
        b=cell(1,M);
        rAux=R;
        for i=1:M
            W{1,i}=generacionW(S(i,1),R);
            b{1,i}=generacionBias(S(i,1));
            R=S(i,1);
        end
        R=rAux;
    else
        W=cell(1,M);
        b=cell(1,M);
        for i=1:M
            textW="W"+num2str(i)+".txt";
            textb="b"+num2str(i)+".txt";
            W(1,i)=num2cell(load(textW));
            b(1,i)=load(textb);
        end
    end
    %Dividimos los datos
    [ptrain,ttrain,pval,tval,ptest,ttest]=separarDatos(p,targets,opcDatos);
    
    
    %Tamanio de p----------------------------------------------------------
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
                %Se guarda el último valor de los pesos y bias
                for i=1:M
                    
                end
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
    %clear;
