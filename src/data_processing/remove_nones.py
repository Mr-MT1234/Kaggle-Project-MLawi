from .data_processor import DataProcessor
import pandas as pd

class RemoveNones(DataProcessor):
    def __init__(self, columns):
        self.columns = columns

    def process_data(self, df:pd.DataFrame):
        to_remove = df[(df[self.columns].isnull()).any(axis=1)].index
        return df.drop(index=to_remove)