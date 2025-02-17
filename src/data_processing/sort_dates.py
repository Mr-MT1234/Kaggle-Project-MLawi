from .data_processor import DataProcessor
import pandas as pd
import numpy as np

class SortDates(DataProcessor):
    def process_data(self, df:pd.DataFrame):
        dates = df[['date0', 'date1', 'date2', 'date3', 'date4']].apply(lambda x: pd.to_datetime(x, dayfirst=True))
        change_statuses = df[['change_status_date0', 'change_status_date1', 'change_status_date2', 'change_status_date3', 'change_status_date4']]
        red_means = df[['img_red_mean_date1', 'img_red_mean_date2', 'img_red_mean_date3', 'img_red_mean_date4', 'img_red_mean_date5']]
        red_stds = df[['img_red_std_date1', 'img_red_std_date2', 'img_red_std_date3', 'img_red_std_date4', 'img_red_std_date5']]
        green_means = df[['img_green_mean_date1', 'img_green_mean_date2', 'img_green_mean_date3', 'img_green_mean_date4', 'img_green_mean_date5']]
        green_stds = df[['img_green_std_date1', 'img_green_std_date2', 'img_green_std_date3', 'img_green_std_date4', 'img_green_std_date5']]
        blue_means = df[['img_blue_mean_date1', 'img_blue_mean_date2', 'img_blue_mean_date3', 'img_blue_mean_date4', 'img_blue_mean_date5']]
        blue_stds = df[['img_blue_std_date1', 'img_blue_std_date2', 'img_blue_std_date3', 'img_blue_std_date4', 'img_blue_std_date5']]

        sorted_indeces = np.argsort(dates.values, axis=1)

        dates_reordered = pd.DataFrame(np.take_along_axis(dates.values, sorted_indeces, axis=1), index=df.index).add_prefix('date')
        change_statuses_reoreded = pd.DataFrame(np.take_along_axis(change_statuses.values, sorted_indeces, axis=1), index=df.index).add_prefix("change_status_date")
        red_means_reoreded = pd.DataFrame(np.take_along_axis(red_means.values, sorted_indeces, axis=1), index=df.index).rename(columns=lambda x: f"img_red_mean_date{int(x)+1}")
        red_stds_reoreded = pd.DataFrame(np.take_along_axis(red_stds.values, sorted_indeces, axis=1), index=df.index).rename(columns=lambda x: f"img_red_std_date{int(x)+1}")
        green_means_reoreded = pd.DataFrame(np.take_along_axis(green_means.values, sorted_indeces, axis=1), index=df.index).rename(columns=lambda x: f"img_green_mean_date{int(x)+1}")
        green_stds_reoreded = pd.DataFrame(np.take_along_axis(green_stds.values, sorted_indeces, axis=1), index=df.index).rename(columns=lambda x: f"img_green_std_date{int(x)+1}")
        blue_means_reoreded = pd.DataFrame(np.take_along_axis(blue_means.values, sorted_indeces, axis=1), index=df.index).rename(columns=lambda x: f"img_blue_mean_date{int(x)+1}")
        blue_stds_reoreded = pd.DataFrame(np.take_along_axis(blue_stds.values, sorted_indeces, axis=1), index=df.index).rename(columns=lambda x: f"img_blue_std_date{int(x)+1}")

        columns = ['date0', 'date1', 'date2', 'date3', 'date4',
'change_status_date0', 'change_status_date1', 'change_status_date2', 'change_status_date3', 'change_status_date4',
'img_red_mean_date1', 'img_red_mean_date2', 'img_red_mean_date3', 'img_red_mean_date4', 'img_red_mean_date5',
'img_red_std_date1', 'img_red_std_date2', 'img_red_std_date3', 'img_red_std_date4', 'img_red_std_date5',
'img_green_mean_date1', 'img_green_mean_date2', 'img_green_mean_date3', 'img_green_mean_date4', 'img_green_mean_date5',
'img_green_std_date1', 'img_green_std_date2', 'img_green_std_date3', 'img_green_std_date4', 'img_green_std_date5',
'img_blue_mean_date1', 'img_blue_mean_date2', 'img_blue_mean_date3', 'img_blue_mean_date4', 'img_blue_mean_date5',
'img_blue_std_date1', 'img_blue_std_date2', 'img_blue_std_date3', 'img_blue_std_date4', 'img_blue_std_date5']

        return df.drop(columns=columns).join(dates_reordered) \
            .join(change_statuses_reoreded) \
            .join(red_means_reoreded).join(red_stds_reoreded) \
            .join(green_means_reoreded).join(green_stds_reoreded) \
            .join(blue_means_reoreded).join(blue_stds_reoreded) 
                