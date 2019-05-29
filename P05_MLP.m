function [] = P05_MLP()
%P5: Multilayer Perceptron
%Fecha de elaboracion: 2019/05/05
%Autor: Villegas Hernandez Carlos Uriel
%Autor: Morales Flores Victor Leonel
%Autor: Flores Garcia Alberto
%Asignatura: Neural Networks
%Escuela: ESCOM-IPN(MX)
    %Variable para almacenar el error de epoca
    
    Eepoch=0;
    %wStoping=fopen('WStoping.txt','w');%txt donde se guardaran los valores de W del stoping
    eStoping=fopen('EStoping.txt','w');%txt donde se guardaran los valores del error de epoca
    %bStoping=fopen('BStoping.txt','w');%txt donde se guardaran los valores de B del stoping Earliny
    %Cont_error=-1;%lleva el conteo de incremento de error en la epoca de validacion
    %Aux_error=100;%guardara el error de la epoca de validacion anterior
    archivoP=input('Ingrese el numero del polinomio deseado: ','s');
    %archivoT=input('Ingrese el nombre del archivo que contiene los valores deseados[targets](sin extension .txt): ','s');
    %rangoInf= input('Indica el valor mínimo en el rango de la señal: ');
    %rangoSup= input('Indica el valor mínimo en el rango de la señal: ');
    fprintf("\nLas opciones de separacion de datos son:");
    fprintf("\n 1)80%% Entrenamiento, 10%% Validacion, 10%% Prueba ");
    fprintf("\n 2)70%% Entrenamiento, 15%% Validacion, 15%% Prueba ");
    opcDatos= input('\nIndica la forma de separar los datos que deseas: ');
    %archivoArq1=input('Ingrese el nombre del archivo que contiene la arquitectura[V1](sin extension .txt): ','s');
    %archivoArq2=input('Ingrese el nombre del archivo que contiene la arquitectura[V2](sin extension .txt): ','s');
    tamarch=input('Ingrese el tamaño del vector 1 de la arquitectura: ');
    Vector1=zeros(1,tamarch);
    Vector2=zeros(1,tamarch-1);
    disp('Vector 1:')
    for i=1:tamarch
        Vector1(i,1)=input('');
    end
    disp('Vector 2: ')
    for i=1:tamarch-1
        Vector2(i,1)=input('');
    end
    alfa= input('Indica el factor de aprendizaje: ');
    eepoch_max= input('Indica el máximo de épocas: ');
    eepoch= input('Indica el error máximo tolerable: ');
    eepoch_val= input('Indica cada cuánto será la época de validacion: ');
    num_val= input('Indica el número máximo de intentos del error de validacion:');
    archi=input('¿Usará archivo para pesos y bias? 1-Si. 0-No: ');
    [p,targets]=lecturaDataSet(archivoP);
    %[R,S,func] = lecturaVectores(archivoArq1,archivoArq2);
    %Aquí se seleccionan los valores que se agarraban en la función lecturaDataset();
    R=Vector1(1,1);
    S=Vector1(2:end,1);
    func=Vector2(1:end,1);
    %Generacion aleatoria de pesos y bias----------------------------------
    [fS,cS]=size(S);
    %Numero de capas
    M=fS;
    if archi==0
        epoca=0;
        W=cell(1,M);
        b=cell(1,M);
        rAux=R;
        for i=1:M
            W{1,i}=generacionW(S(i,1),R);
            b{1,i}=generacionBias(S(i,1));
            R=S(i,1);
        end
        R=rAux;
        GuardarArchivo(epoca,W,b,M,"w");
        GuardarEepoch(epoca,Eepoch,"w",1);
        GuardarEepoch(epoca,Eepoch,"w",0);
    else
        [W,b,epoca] = RecuperarDatos(M,R,S);
    end
    
    %Dividimos los datos
    [ptrain,ttrain,pval,tval,ptest,ttest]=separarDatos(p,targets,opcDatos);
    %%%%%%%%%%%%%%%%%%%%%%%%%
    WStopping=cell(1,M);
    BStopping=cell(1,M);
    %%%%%%%%%%%%%%%%%%%%%%%%%
    
    %Tamanio de p
    [fptrain,cptrain]=size(ptrain);
    [fpval,cpval]=size(pval);
    [fptest,cptest]=size(ptest);
    
    %Propagacion
    ErrorEarly=-1;
    EarlyStopping=0;
    for k=1:eepoch_max
        epoca=epoca+1;
        if EarlyStopping==num_val
            fprintf('\nSe activa EARLY-STOPPING!!!');
            W=WStopping;
            b=BStopping;
            GuardarArchivo(epoca,W,b,M,"a");
            break;
        end
        fprintf(strcat("\nEpoca ",int2str(epoca),":"));
        Eepoch=0;
        
        if(mod(k, eepoch_val)==0)
           fprintf("\n***Iniciando Validacion***");
            for valp=1:fpval
                a=propagacionAdelante(M,W,b,pval(valp,1),func);
                [e,he]=errorAprendizaje(tval(valp,1),a{1,M});
                Eepoch=Eepoch+abs(e);
            end
            Eepoch=Eepoch/fpval;
            %Sustituye imprimeStoping a fprintf?
            fprintf("\n>>>>>Error de epoca %d: %f",epoca,Eepoch);
            GuardarEepoch(epoca,Eepoch,"a",1);
            ImprimirStoping(Eepoch,eStoping);%Guarda el valor de error de epoca el cual se graficara
            %verifica si si hay incremento en el error  de epoca de validacion
            if ErrorEarly==-1
                ErrorEarly=Eepoch;
                WStopping=W;
                BStopping=b;
            elseif Eepoch>ErrorEarly
                EarlyStopping=EarlyStopping+1;
            else
                ErrorEarly=Eepoch;
                EarlyStopping=0;
                WStopping=W;
                BStopping=b;
            end
                   
        else
            %Entrenamiento
            fprintf("\n---Entrenamiento---");
            for valp=1:fptrain
                a=propagacionAdelante(M,W,b,ptrain(valp,1),func);
                %M porque ahi se encuentra la salida de la ultima capa de
                %la red
                [e,he]=errorAprendizaje(ttrain(valp,1),a{1,M});
                %abs de e?
                Eepoch=Eepoch+abs(e);
                [W,b] =Backpropagation(W,b,a,M,func,e,alfa,ptrain(valp,1));
            end 
            Eepoch=Eepoch/fptrain;
            GuardarArchivo(epoca,W,b,M,"a");
            GuardarEepoch(epoca,Eepoch,"a",0);
            %Falta guardar pesos y bias de cada epoca
            fprintf("\n>>>>>Error de epoca %d: %f",epoca,Eepoch);
            if(Eepoch<eepoch )
                fprintf("\n>>>>>>El valor de error de la red es menor al error tolerable. Acaba entrenamiento");
                break;
            end
        end
    end
    GraficarEepoch(1);
    fprintf("\n---Prueba---");
    %size(ptest)
    %size(ttest)
    %ttest
    Eepoch=0;
    y=zeros(fptest,1);
    for valp=1:fptest
        a=propagacionAdelante(M,W,b,ptest(valp,1),func);
        y(valp,1)=a{1,M};
        %M porque ahi se encuentra la salida de la ultima capa de
        %la red
        [e,he]=errorAprendizaje(ttest(valp,1),a{1,M});
        Eepoch=Eepoch+abs(e);
        %fprintf("\n a: %f | target: %f | error: %f",a{1,M},ttest(valp,1),e);
        
    end 
    Eepoch=Eepoch/fptest;
    fprintf("\nError de epoca de prueba:%f",Eepoch);
    %Visualizacion del polinomio a tratar
    GraficarPolinomio(p,targets,ptest,y);
    %WStopping=W;
    %BStopping=b;
    %epocaStopping=epoca;
    epoca
    GraficarEvolucion(M,R,S);
end
