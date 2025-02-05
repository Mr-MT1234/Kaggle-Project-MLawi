import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from enum import Enum

class ChangeTypes(Enum):
    DEMOLITION = 0
    ROAD = 1
    RESIDENTIAL = 2
    COMMERCIAL = 3
    INDUSTRIAL = 4
    MEGA_PROJECTS = 5


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
