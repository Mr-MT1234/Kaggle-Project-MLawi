from abc import ABC, abstractmethod
import numpy as np

class Classifier(ABC):
    
    @abstractmethod
    def train(self, data: np.ndarray, labels: np.ndarray, weights: np.ndarray = None):
        ...

    @abstractmethod
    def predict(self, input: np.ndarray) -> np.ndarray:
        ...