import abc
import h5py
import numpy as np
import os
from prediction_data import PredictionData


class SummaryEvaluator(PredictionData):

    def __init__(self, ds_name, model_name='run_1_two', num_classes=2):
        super().__init__(ds_name=ds_name, model_name=model_name, num_classes=num_classes)
        self.ds_name = ds_name
        self.model_name = model_name
        self.num_classes = num_classes
        self.class_names = np.asarray(['not in summary', 'in summary'])


    @abc.abstractmethod
    def get_metrics(self, random=False, threshold=None):
        y = self.get_y(random, threshold)

    @abc.abstractmethod
    def plot(self):
        pass

