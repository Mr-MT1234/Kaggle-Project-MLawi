from .data_processor import DataProcessor
import pandas as pd



class MapLabels(DataProcessor):
    def process_data(self, df: pd.DataFrame):
        change_type_map = {'Demolition': 0, 'Road': 1, 'Residential': 2, 'Commercial': 3, 'Industrial': 4,
            'Mega Projects': 5}
        return df.assign(change_type=df.change_type.apply(lambda x: change_type_map[x]))