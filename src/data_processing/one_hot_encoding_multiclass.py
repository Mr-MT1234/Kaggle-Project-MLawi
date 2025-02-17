from .data_processor import DataProcessor
import pandas as pd

class OneHotEncodingMulticlass(DataProcessor):
    def __init__(self, column):
        self.column = column

    def process_data(self, df:pd.DataFrame):
        encoded_df =df[self.column].str.get_dummies(sep=',').add_prefix(self.column+"_")

        return df.drop(columns=self.column).join(encoded_df)