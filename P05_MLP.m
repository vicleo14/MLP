function [] = P05_MLP()
%P5: Multilayer Perceptron
%Fecha de elaboraci�n: 2019/05/05
%Autor: Villegas Hern�ndez Carlos Uriel
%Autor: Morales Flores Victor Leonel
%Asignatura: Neural Networks
%Escuela: ESCOM-IPN(MX)

    archivoP=input('Ingrese el nombre del archivo que contiene las entradas[p](sin extension .txt): ','s');
    archivoT=input('Ingrese el nombre del archivo que contiene los valores deseados[targets](sin extension .txt): ','s');
    rangoInf= input('Indica el valor m�nimo en el rango de la se�al: ');
    rangoSup= input('Indica el valor m�nimo en el rango de la se�al: ');
    fprintf("\nLas opciones de separaci�n de datos son:");
    fprintf("\n 1)80% Entrenamiento, 10% Validaci�n, %10% Prueba ");
    fprintf("\n 2)70% Entrenamiento, 15% Validaci�n, %15% Prueba ");
    opcDatos= input('Indica la forma de separar los datos que deseas: ');
    archivoArq1=input('Ingrese el nombre del archivo que contiene la arquitectura[V1](sin extension .txt): ','s');
    archivoArq2=input('Ingrese el nombre del archivo que contiene la arquitectura[V2](sin extension .txt): ','s');
    alfa= input('Indica el factor de aprendizaje: ');
    eepoch_max= input('Indica el m�ximo de �pocas: ');
    e_epoch= input('Indica el error m�ximo tolerable: ');
    eepoch_val= input('Indica cada cu�nto ser� la �poca de validaci�n: ');
    num_val= input('Indica el n�mero m�ximo de intentos del error de validaci�n:');
end

