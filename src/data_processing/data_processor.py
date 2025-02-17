from abc import ABC, abstractmethod
import pandas as pd

class DataProcessor(ABC):
     
    @abstractmethod
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        ...

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.process_data(data)