# -*- coding: utf-8 -*-
"""
    Uses multiple detectors as an ensemble. These can each be better suited to detect different types of drift, and their outputs analysed to recognise multiple different types of drift, thus mitigating the weaknesses each individual method might have (pending testing)
"""
import numpy as np
import capymoa.drift.detectors as detectors
from typing import Any, Dict
import capymoa.stream.generator as streamGen
from capymoa.drift.base_detector import BaseDriftDetector
from capymoa.stream import Stream, Schema, NumpyStream
from capymoa.instance import Instance, LabeledInstance
from capymoa.classifier import OzaBoost, MajorityClass, NaiveBayes
from capymoa.evaluation import prequential_evaluation
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import CategoricalNB
from collections import defaultdict
import itertools
import collections

class EnsembleDetector(BaseDriftDetector): #Inheriting from BaseDriftDetector of MOADriftDetector would require the warnings and drift detections be returned as booleans, 
    def __init__(self, 
                 #datasetSize : int,
                 #dataset : Stream, 
                 detectorDict=[("ADWIN",{}),
                               ("CUSUM",{}),
                               ("DDM",{}),
                               #("EWMAChart",{}),
                               ("HDDMAverage",{}),
                               ("HDDMWeighted",{})],
                 valid_delay: int = 500,
                 cache=True
                 ):
        detectorSet=[]
        detectorNames=[]
        schemaFeatures=[]
        for detector_name, detector_args in detectorDict:
            detectorSet.append(getattr(detectors, detector_name)(**detector_args))
            detectorNames.append(detector_name)
            schemaFeatures.append(detector_name)
        self.baseDetectorNameList=detectorNames
        self.numberOfDetectors=len(self.baseDetectorNameList)
        self.detection_index=[]
        self.resultsDictionary=[]
        #print(self.baseDetectorNameList)
        self.baseDetectorsList=detectorSet
        self.idx=0
        self.cacheResults=cache
        self.driftDetectionModel=None
        #print(self.baseDetectorNameList)
        #print(self.baseDetectorNameList)

        schemaFeatures.append("Drift Detected")
        #print(self.baseDetectorNameList)
        #print(type(detectorNames))
        #self.streamSchema=Schema.from_custom(features=schemaFeatures, target="Drift Detected", categories={"Drift Detected":["0","1"]}, name="Drift Detection Instances")
        #print(self.streamSchema, self.streamSchema.is_classification())
        self.validDelay=valid_delay
        self.maxsWindow=np.array([[0, self.validDelay] for detector in self.baseDetectorNameList])
        self.in_concept_change=False
        self.frequencies={}
        #print(self.training_data)
        #self.training_report=[]
        #print("Iniiaospreentrenaiento",flush=True)
        #self.pretrain()
        #print("Iniiaosdescripion",flush=True)
        #self.describe_drifts()
    
    def feed_training_data(self,dataset_size, dataset):
        self.datasetSize=dataset_size
        self.training_data : Stream=dataset
    
    def get_params(self) -> Dict[str, Any]:
        """Get the hyper-parameters of the drift detector."""
        return {
            "baseDetectorsList": self.baseDetectorsList,
            "idx": self.idx,
            "training_data":self.training_data,
            "training_report":self.training_report
        }
    def store_results(self)->None:
        #print("ww",flush=True)
        outputs=[0.0,1.0,2.0]
        outputsCartesianProduct=itertools.product(outputs, repeat=len(self.baseDetectorNameList))
        dictResults={}
        j=0
        for reading in outputsCartesianProduct:
            #reading=np.array(reading, dtype=int)
            #print(reading)
            instance=Instance.from_array(self.training_report.get_schema(), np.array(reading))
            prediction=self.driftDetectionModel.predict(instance)
            
            #if (self.frequencies[reading]==0) :
                # print(reading,self.frequencies[reading])
            #    j+=1
            # print("Resultados del clasificador",reading, prediction, flush=True)
            #if prediction is None:
            #    prediction=self.__get_lower_results__(reading, dictResults)
            dictResults[reading]=prediction
        # i=0
        correctedDict=self.__get_lower_results__(dictResults)
        # for reading, value in dictResults.items():
        #     if (value == 1):
        #         i+=1
        #         print("No value found",reading ,i,flush=True)
        #     if(dictResults[reading]==1):
        #         print("Resultados en el diccionario",reading, dictResults[reading], flush=True)
        print("Número y fracción de estados no visitados",len(self.frequencies),len(self.frequencies)/(3**(self.numberOfDetectors)),flush=True)
        self.resultsDictionary=correctedDict
        #print(correctedDict)
    def __get_lower_results__(self,dictionary):
        newDictionary=dictionary
        for key, value in dictionary.items():
            readingItemCounter=list(key)
            if value==0:
                #print("Centro de recursión",readingItemCounter, flush=True)
                for i in range(self.numberOfDetectors):
                    if readingItemCounter[i]!=0:
                        readingItemCounter[i]-=1
                        #print(readingItemCounter)
                        readingNew=tuple(readingItemCounter)
                        #print("Resultados en recursión",readingNew,dictionary[readingNew], flush=True)
                        if dictionary[readingNew]==1:
                            newDictionary[key]=1
                            break
                #:
                    
                    #readingItemCounter[i]+=1
                    #readingNew=tuple(readingList)
                    #return self.__get_lower_results__(readingNew, dictionary)
        return newDictionary
    
    def add_element(self, element : float)->None:
        """Update each of the base detectors with a new input value.

        :param element: A value to update the drift detector with. Usually,
            this is the prediction error of a model.
        """
        if self.detected_change():
            self.reset()
        self.idx+=1
        for detector in self.baseDetectorsList:
            detector.add_element(element)
        reading=self.get_states()
        
        for j in range(len(reading)):
            if reading[j]>=self.maxsWindow[j][0] or self.maxsWindow[j][1]==0:
                self.maxsWindow[j][0]=reading[j]
                self.maxsWindow[j][1]=self.validDelay
            else:
                self.maxsWindow[j][1]-=1
        maxReading=tuple([self.maxsWindow[i,0] for i in range(len(self.baseDetectorNameList))])
        if self.cacheResults:
            #print("Máximos como array",self.maxsWindow[0][:])
            #print("Máximos como tupla",maxReading)
            prediction=self.resultsDictionary[maxReading]
            # if prediction==1:
            #     print("Máximos como tupla",maxReading)
            #     print("Predicción", prediction, flush=True)
        else:
            instance=Instance.from_array(self.training_report.get_schema(), #[self.maxsWindow[i][0] for i in range(len(self.baseDetectorNameList))]
                                    np.array([self.maxsWindow[i][0] for i in range(len(self.baseDetectorNameList))]))
            prediction=self.driftDetectionModel.predict(instance)
        #instance=np.reshape(instance, shape=(1,-1))
        #print(instance)
        #print(self.idx)
        #print(self.driftDetectionModel.predict(instance))
        if(prediction):
            #print("De detecta cambio")
            self.in_concept_change=True
            self.detection_index.append(self.idx)
            #print("Se debe reiniciar")
            #self.reset()
            
        
    
    def __add_element_pre(self, element : float)->None:
        """Update each of the base detectors with a new input value.

        :param element: A value to update the drift detector with. Usually,
            this is the prediction error of a model.
        """
        for detector in self.baseDetectorsList:
            detector.add_element(element)
        #instance=Instance.from_array(self.training_report.get_schema(), np.array(self.get_states()))
        #print(instance)
        #print(self.driftDetectionModel.predict(instance))
        #self.idx+=1
    
    def get_states(self)->[int]:
        """Returns whether each of the detectors is detecting change (2), is issuing a warning (1), or none of those (0). 
        """
        states=[]
        for detector in self.baseDetectorsList:
            if (detector.detected_change()):
                states.append(2.0)
            elif(detector.detected_warning()):
                states.append(1.0)
            else:
                states.append(0.0)
        return np.array(states, dtype=int)
    
    def get_all_base_warnings(self) ->[[int]]:
        """Returns the warning indexes of the base detectors
        """
        base_warnings=[]
        for detector in self.baseDetectorsList:
            base_warnings.append(detector.warning_index)
        return base_warnings
    
    def get_all_base_detections(self)-> [[int]]:
        """Returns the detection indexes of the base detectors
        """
        base_detections=[]
        for detector in self.baseDetectorsList:
            base_detections.append(detector.detection_index)
        return base_detections
    
    def get_all_detections(self)->[int]:
        return self.detection_index
    
    def detected_change(self)-> bool :
        #print("Compropbando cambio")
        return self.in_concept_change
    
    def reset(self, clean_history: bool = False) -> None:
        """Reset the ensemble training data.

        :param clean_history: Whether to reset detection history, defaults to False
        """
        #print("Resetting")
        for detector in self.baseDetectorsList:
            detector.reset()
        self.in_concept_change=False
        self.maxsWindow=np.array([[0, self.validDelay] for i in range(len(self.baseDetectorNameList))])
        if clean_history:
            #self.maxsWindow=[[0, self.validDelay] for detector in self.baseDetectorNameList]
            self.detection_index = []
            self.warning_index = []
            self.idx = 0
            
    def deleteDetections(self):
        self.detection_index = []
        self.warning_index = []
        self.idx = 0   
        
    def preprocess_training_data(self):
        #dataSchema : Schema=self.training_data.get_schema()
        #print(self.training_data.next_instance())
        #print(dataSchema, dataSchema.get_label_indexes())
        """ Receives a dataset that should include multiple types of drift. The dataset must be synthetic, as the location for every concept drift and its type must be known beforehand. The states of the ensemble are saved in training_report"""
        #detection_history=np.empty
        maxTrue=[0,self.validDelay]
        trueDrifts=np.empty(shape=[self.datasetSize, 1], dtype=np.int8)
        maxsDict=np.array([[0,self.validDelay] for detector in range(len(self.baseDetectorNameList))],dtype=np.int16)
        allDetections=np.empty(shape=[self.datasetSize, len(self.baseDetectorsList)],dtype=np.int8)
        i=0
        #k=0
        for instance  in self.training_data:
            # print(instance)
            # if i<4:
            #     print(i)
            flagWipe=False
            exitConceptDetectionWindow=False
            #print(instance.schema, instance)
            #print(i)
            #print(instance.y_label)
            #print(instance)
            y=int(instance.y_label)
            
            if maxTrue[0]==1 and maxTrue[1]==0:
                exitConceptDetectionWindow=True
            if y>=maxTrue[0] or maxTrue[1]==0:
                maxTrue[0]=y
                maxTrue[1]=self.validDelay
            else:
                maxTrue[1]-=1
            
            if maxTrue[0]==1:
                flagWipe=True
        
            #print("trueDrifts= ",trueDrifts)
            self.__add_element_pre(instance.x)
            
            instanceDetections=self.get_states()
            # print("Detecciones instantáneas: ",instanceDetections)
            #print("Almacenamiento de ventana: ",maxTrue)
            for j in range(len(instanceDetections)):
                if instanceDetections[j]>maxsDict[j,0] or maxsDict[j,1]==0:
                    maxsDict[j,0]=instanceDetections[j]
                    maxsDict[j,1]=self.validDelay
                else:
                    maxsDict[j,1]-=1
                if exitConceptDetectionWindow:
                    #print("Exit ", i)
                    if maxsDict[j,0]==1:
                       self.baseDetectorsList[j].reset() 
                    maxsDict[j,0]=0
                    maxsDict[j,1]=self.validDelay
                    #if self.baseDetectorsList[j].detected_warning():
                        
                instanceDetections[j]=maxsDict[j,0]
                if (maxsDict[j,0]==0 and flagWipe):
                    flagWipe=True
                else:
                    flagWipe=False
                    
                
                    
            #print("Instance Detections= ", instanceDetections)
            if not flagWipe:
                
                allDetections[i,:]=instanceDetections[:]
                #print(instanceDetections, allDetections[i,:],i)
                trueDrifts[i,0]=maxTrue[0]
                i+=1
                # print("Resultados de los detectores", allDetections[i,:])
                # print("Diccionario de máximos", maxsDict)
                # print("Lectura de cambio aceptable: ",trueDrifts[i,0])
                # print("Eliminación de instancia, salida de cambio de concepto",flagWipe, exitConceptDetectionWindow)
            #else:
                #k+=1
                #print(instance, maxTrue,instanceDetections)
                #print("Purged element number ", i)
            #print("All detections= ", allDetections)
            
            #Pasa las etiquetas como list, los resultados de los detectores como array de arrays (vease NumpyArray)
            # print("Cambio", self.training_data[1,i])
            # instance=self.get_states()
            # instance.append(self.training_data[1,i])
            # print("Instancia de entrenamiento:",instance)
            # detection_history[i]=instance
        #Si el delay aceptado es bajo (<MTTD) y los cambios muy frecuentes, pueden etiquetarse las instancias donde ningún sensor detecte cambio, como instancias con cambio
        #detectionHistory=(allDetections, np.reshape(trueDrifts, shape=self.datasetSize))
        #allDetections=allDetections[:i,:]
        purgedDetections=allDetections[:i]
        purgedTrueDrifts=trueDrifts[:i]
        
        del allDetections
        del trueDrifts
        
        detectionHistory : NumpyStream[LabeledInstance]=NumpyStream(purgedDetections, 
                                                                    purgedTrueDrifts, 
                                                                    dataset_name='detectionStream', 
                                                                    feature_names=self.baseDetectorNameList, 
                                                                    target_name='Ground-truth', 
                                                                    target_type='categorical')
        #expandedReading=np.concatenate((purgedDetections, purgedTrueDrifts), axis=1)
        #print(expandedReading)
        self.frequencies=collections.Counter(self.frequencies)+collections.Counter(map(tuple,purgedDetections))
        #print(self.frequencies)
        #print(detectionHistory.get_schema())
        # l=0
        # for instance in detectionHistory:
        #     print(allDetections[l,:])
        #     print(detectionHistory.next_instance())
        #     print(l)
        #     l+=1
        #print(allDetections)
        #print(detectionHistory)
        self.training_report=detectionHistory 
        #print(k)
        
    def trainClassifier(self):
        #print(self.training_report.get_schema())
        #Usar uno de los tres algoritmos online de capymoa, tomando anteriormente el máximo en ventana
        #weakLearner=CategoricalNB()
        #classifier = AdaBoostClassifier(n_estimators=10,estimator=weakLearner)
        #classifier.fit(self.training_report[0], self.training_report[1])
        classifier=OzaBoost(self.training_report.get_schema()
                            ,base_learner="bayes.NaiveBayes"
                              )
        # #print(self.training_report.get_schema())
        i=0
        #outputs=[0.0,1.0,2.0]
        #outputsCartesianProduct=itertools.product(outputs, repeat=len(self.baseDetectorNameList))
        #dictFrequencies = defaultdict(int)
        for instance in self.training_report:
        #     #print(instance.y_label)
            
        #     #print(i)
            

            # probs=classifier.predict_proba(instance)
            # pred=classifier.predict(instance)
            # if probs is not None:
            #     if probs.size==1:
            #         probsVector=np.zeros(2)
            #         probsVector[pred]=probs
            #         probs=probsVector
            #     if instance.y_index==1:
            #         if probs[instance.y_index]<0.5:
            #             print("Clasificación errónea en i=",i," con lecturas=",instance.x, ", probabilidad=", probs[instance.y_index], "para el valor correcto", instance.y_index," y predicción ",pred )
            #         else:
            #             print("Clasificación correcta en i=",i," con lecturas=",instance.x, ", probabilidad=", probs[instance.y_index], "para el valor correcto", instance.y_index )
            #else:
            #    print(instance)
            #    print("Clasificación anómala en i=",i," con lecturas=",instance.x, "para el valor correcto", instance.y_index," y predicción ",pred )
            
        #     #if(i<=1211000):
            classifier.train(instance)
            #dictFrequencies[tuple(instance.x)]+=1
            i+=1
        #print(i)
        #self.training_report
        #results = prequential_evaluation(self.training_report, classifier, restart_stream=True)
        #print(results.predictions())
        #self.frequencies=dictFrequencies
        self.driftDetectionModel=classifier
        #print("Lasifi",flush=True)
        if self.cacheResults:
            self.store_results()
        #print(results.cumulative)


