import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

CHANGE_TYPE_DEMOLITION = 0
CHANGE_TYPE_ROAD = 1
CHANGE_TYPE_RESIDENTIAL = 2
CHANGE_TYPE_COMMERCIAL = 3
CHANGE_TYPE_INDUSTRIAL = 4
CHANGE_TYPE_MEGA_PROJECTS = 5


class Model(ABC):
    @abstractmethod
    def preprocess_dataset(self, dataset : pd.DataFrame):
        pass

    @abstractmethod
    def train(self, dataset):
        pass
    
    @abstractmethod
    def predict(self, input) -> pd.DataFrame:
        pass