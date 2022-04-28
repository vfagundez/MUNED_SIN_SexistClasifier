# SexistClasifier

Este es el codigo del proyecto final de la asignatura Sistemas de
información no estructura del Master Universitario en Ingeniería
Informática.

El objetivo de este proyecto es generar un sistema de clasificación
que dado un tweet determine si contiene contenido sexista o no, tal y como 
se describe en la Tarea 1 de [EXIST 2022](http://nlp.uned.es/exist2022/), 
a partir del texto del tweet, y para los corpus proporcionados 
por la organización.


## Comenzando 🚀

_Estas instrucciones te permitirán obtener una copia del proyecto en funcionamiento en tu máquina local para propósitos de desarrollo y pruebas._


### Pre-requisitos 📋
Es necesario tener instalada la libreria _Pandas_. En el caso de que no lo
tengas instalado, _Pandas_ se distribuye a
través de _pip_ como una _wheel_, lo que significa que debes ejecutar los 
siguientes comandos en tu terminal.

```
pip install wheel
pip install pandas
```
Tambien debemos tener instalada la libreria _spacy_ y _sklearn_
```
pip install spacy
pip install sklearn
```
Finalmente instalamos el modulo _es_core_news_sm_ de spacy
```
python -m spacy download es_core_news_sm
```
Aunque he tenido problemas con la instalación de este modulo debido a la ruta
de instalación de mi python, y he tenido que utilizar el siguiente comando:
```
'C:\Users\Victor\AppData\Local\Programs\Python\Python310\python.exe' -m spacy download es_core_news_sm
```
### Instalación y Configuración 🔧

Una vez descargado el proyecto encontramos la siguiente estructura de carpetas
```
.
├── data
├── docs
├── etc
├── logs
├── src
└── readme.md
```    
* en la carpeta _data_ encontraremos los archivos con los datos de entrenamiento
 y test
* en la carpeta _docs_ encontraremos los archivos de configuración
* en la carpeta _etc_ encontraremos los archivos de configuración
* en la carpeta de _logs_ encontraremos todos los archivos de registro de
la aplicación
* en la carpeta _src_ encontramos el codigo fuente de la aplicación


## Funcionamiento ⚙️

Una vez, hemos completado los pasos mostrados en la sección de Prerequisitos 
[Prerequisitos](#pre-requisitos_📋)


## Autores ✒️

* **Jorge Carrillo de Albornoz** - *Clasificador basico*
* **Victor Fagúndez Poyo** - *Modificaciones al algoritmo base* - [vincitori-dev](https://github.com/vincitori-dev)



