from .data_processor import DataProcessor
import pandas as pd

class DifferencesInDates(DataProcessor):
    def process_data(self, df: pd.DataFrame):
        dates = [pd.to_datetime(df[d], dayfirst=True) for d in ['date0', 'date1', 'date2', 'date3', 'date4']]
        df = df.copy(deep=True)
        df['date_delta1'] = (dates[1] - dates[0]).apply(lambda x: x.days)
        df['date_delta2'] = (dates[2] - dates[0]).apply(lambda x: x.days)
        df['date_delta3'] = (dates[3] - dates[0]).apply(lambda x: x.days)
        df['date_delta4'] = (dates[4] - dates[0]).apply(lambda x: x.days)

        return df.drop(columns=['date0', 'date1', 'date2', 'date3', 'date4'])
