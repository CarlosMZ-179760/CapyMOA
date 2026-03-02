# -*- coding: utf-8 -*-
"""
    Uses multiple detectors as an ensemble. These can each be better suited to detect different types of drift, and their outputs analysed to recognise multiple different types of drift, thus mitigating the weaknesses each individual method might have (pending testing)
"""
import capymoa.drift.detectors as detectors
from typing import Any, Dict

class EnsembleDetector: #Inheriting from BaseDriftDetector of MOADriftDetector would require the warnings and drift detections be returned as booleans, 
    def __init__(self, detectorDict=[("HDDMAverage", {}),("CUSUM", {}),("ADWIN", {})]): #Would it be preferable to pass the detector list as an argument directly?
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
            "idx": self.idx,
            "training_data":self.training_data,
            "training_report":self.training_report
        }
    
    def add_element(self, element : float)->None:
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
    
    def reset(self, clean_history: bool = False) -> None:
        """Reset the ensemble training data.

        :param clean_history: Whether to reset detection history, defaults to False
        """
        
        self.training_data=[]
        self.training_report=[]

        if clean_history:
            self.detection_index = []
            self.warning_index = []
            self.idx = 0
            
    def pretrain(self, dataset : [float]):
        """ Receives a dataset that should include multiple types of drift. The dataset mst be synthetic, as the location for every concept drift and its type must be known beforehand. The states of the ensemble are saved in training_report"""
        self.training_data=dataset
        detection_history=[]
        for datapoint in self.training_data:
            self.add_element(datapoint)
            detection_history.append(self.get_states())
        self.training_report=detection_history
    
    def describe_drifts(self):
        """ Using training_report and the known drift locations in training_data, this method should extract which states of the ensemble correspond to different types of drift, or which can be ignored. This way, different types of drift could be characterized depending on their corresponding states."""
        raise NotImplementedError
