# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 12:54:58 2026

@author: Carlos Martínez Zurita
"""
import capymoa.drift.detectors as detectors
from typing import Any, Dict

class EnsembleDetector: #Inheriting from BaseDriftDetector of MOADriftDetector would require the warnings and drift detections be returned as booleans, 
    def __init__(self, detectorDict=[("HDDMAverage", {}),("CUSUM", {}),("ADWIN", {})]):
        detectorSet=[]
        for detector_name, detector_args in detectorDict:
            detectorSet.append(getattr(detectors, detector_name)(**detector_args))
        self.baseDetectorsList=detectorSet
        self.idx=0
        self.training_data=[]
        self.training_report=[]
    def get_params(self) -> Dict[str, Any]:
        """Get the hyper-parameters of the drift detector."""
        return {
            "baseDetectorsList": self.baseDetectorsList,
            "idx": self.idx
        }
    
    def add_element(self, element : float):
        """Update each of the base detectors with a new input value.

        :param element: A value to update the drift detector with. Usually,
            this is the prediction error of a model.
        """
        for detector in self.baseDetectorsList:
            detector.add_element(element)
        self.idx+=1
    
    def get_states(self)->[int]:
        """Returns whether each of the detectors is detecting change (2), is issuing a warning (1), or none of those (0). 
        """
        states=[]
        for detector in self.baseDetectorsList:
            if (detector.detected_change()):
                states.append(1)
            elif(detector.detected_warning()):
                states.append(2)
            else:
                states.append(0)
        return states
    
    def get_all_base_warnings(self):
        base_warnings=[]
        for detector in self.baseDetectorsList:
            base_warnings.append(detector.warning_index)
        return base_warnings
    
    def get_all_base_detections(self):
        base_detections=[]
        for detector in self.baseDetectorsList:
            base_detections.append(detector.detection_index)
        return base_detections
    
    def reset(self, clean_history: bool = False) -> None:
        """Reset the drift detector.

        :param clean_history: Whether to reset detection history, defaults to False
        """
        
        

        if clean_history:
            self.detection_index = []
            self.warning_index = []
            self.idx = 0
            
    def pretrain(self, dataset : [float]):
        self.training_data=dataset
        detection_history=[]
        for datapoint in self.training_data:
            self.add_element(datapoint)
            detection_history.append(self.get_states())
        self.training_report=detection_history
