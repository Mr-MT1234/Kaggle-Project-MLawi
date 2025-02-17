from .data_processor import DataProcessor
import pandas as pd

class OneHotEncoding(DataProcessor):
    def __init__(self, columns):
        self.columns = columns

    def process_data(self, df:pd.DataFrame):
        all_categories = sorted(pd.DataFrame(df[self.columns]).stack().unique())

        for col in self.columns:
            df[col] = pd.Categorical(df[col], categories=all_categories)

        one_hot_encoded = pd.get_dummies(df[self.columns], prefix_sep='_')

        return df.drop(columns=self.columns).join(one_hot_encoded)