from .data_processor import DataProcessor
import pandas as pd

class Replace(DataProcessor):
    def __init__(self, column, dict=None):
        self.column = column
        self.dict = dict

    def process_data(self, df:pd.DataFrame):
        replaced = df[self.column].replace(self.dict)
        return df.drop(columns=self.column).join(replaced)