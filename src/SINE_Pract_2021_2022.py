'''
El script no contiene acentos para evitar problemas de codificacion.

Es importante tener el entorno de desarrollo configurado para utf-8 
o puede dar problemas al leer ciertos caracteres del dataset de EXIST

@author: Jorge Carrillo de Albornoz
'''

import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

#Funcion que dado un documento de Spacy devuelve una lista de palabras.
def getToken(docSpacy):
    listTokens = []
    for token in docSpacy:
        listTokens.append(token.text)
    
    return listTokens

#Funcion tonta que devuelve el documento pasado por parametro.        
def dummy(doc):
    return doc

#Funcion principal para classificar comentarios en base a su sexismo
def processEXISTTraining(pathTraining, pathTest):
    #Leemos los archivos tsv de entramiento
    print("Leyendo EXIST dataset")
    datasetTraining = pd.read_csv(pathTraining, sep="\t", dtype = str, encoding='utf-8')
    datasetTest = pd.read_csv(pathTest, sep="\t", dtype = str, encoding='utf-8')
    frames = [datasetTraining, datasetTest]
    
    #concatenamos los dos conjuntos
    dataset = pd.concat(frames)
    
    #Filtramos solo para el espanol      
    dataset= dataset.loc[dataset['language'] == 'es'] 
    print(dataset.shape)

    #Cargamos el modelo del lenguaje para el espanol
    nlp= spacy.load('es_core_news_sm')
    
    #Procesamos los textos del dataset de EXIST
    print("Procesando comentarios para extraer sus palabras")
    lstDocsEXIST = []
    lstObjetiveClass = []
    index=1    
    for row in dataset.itertuples(index=False):
        #Comentar la siguiente linea si tarda mucho
        print("Generando documento exist "+ str(index) + "; claseObjetiva: " + row.task1 + "; texto: " +row.text )
        lstDocsEXIST.append(getToken(nlp(row.text)))
        lstObjetiveClass.append(row.task1)
        index=index+1
   
    #Generamos la matriz td-idf
    print("Generando matriz de tf-idf")
    tfidf = TfidfVectorizer(tokenizer=dummy, preprocessor=dummy)
    tfidf_matrix = tfidf.fit_transform(lstDocsEXIST)
    lstFeaturesTFIDFByComment = tfidf_matrix.toarray()
    print(len(lstFeaturesTFIDFByComment))
    
    #Configuramos el modelo de SVM para entrenar/predecir
    print("Entrenando el algoritmo SVM")
    clf = svm.SVC(kernel='linear') 
    
    #Generamos los 10 fold para entrenar/predecir, y evaluamos mostrando la accuracy
    scores = cross_val_score(clf, lstFeaturesTFIDFByComment, lstObjetiveClass, cv=10) 
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    
    print("Mostrando metricas mas precisas y matriz de confusion")
    classNames = dataset.task1.unique() 
    classesPredicted = cross_val_predict(clf, lstFeaturesTFIDFByComment, lstObjetiveClass, cv = 10)       
    print(classification_report(lstObjetiveClass, classesPredicted, target_names=classNames))
    print(confusion_matrix(lstObjetiveClass, classesPredicted))
    

if __name__ == '__main__':
    pathTraining = "./data/EXIST2021_training_100.tsv"
    pathTest = "./data/EXIST2021_test_labeled_100.tsv"
    processEXISTTraining(pathTraining, pathTest)
