import pandas as pd
import numpy as np
from datetime import datetime

def clean_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    output = dataset

    # Remove rows that contain None
    for c in output.columns:
        output = output.drop(output.index[output[c].isnull()])

    # Reorder the dates rows to be in chronological order
    def sort_row(row):
        to_date_time = lambda x: datetime.strptime(x, '%d-%m-%Y')
        
        date0 = to_date_time(row.date0), row.change_status_date0
        date1 = to_date_time(row.date1), row.change_status_date1
        date2 = to_date_time(row.date2), row.change_status_date2
        date3 = to_date_time(row.date3), row.change_status_date3
        date4 = to_date_time(row.date4), row.change_status_date4

        s = sorted([date0, date1,date2,date3,date4])

        return {
            'date0' : s[0][0],
            'change_status_date0' : s[0][1],
            'date1' : s[1][0],
            'change_status_date1' : s[1][1],
            'date2' : s[2][0],
            'change_status_date2' : s[2][1],
            'date3' : s[3][0],
            'change_status_date3' : s[3][1],
            'date4' : s[4][0],
            'change_status_date4' : s[4][1],
        }

    datetable = output.apply(sort_row, axis=1, result_type='expand')

    output = output.drop(columns=['date0', 'change_status_date0', 'date1', 'change_status_date1','date2', 'change_status_date2','date3', 'change_status_date3', 'date4', 'change_status_date4'])
    output = output.join(datetable)

    return output

_geography_types_representations = {
    'N,A':              np.zeros(11),
    'Barren Land' :     np.eye(11)[0],
    'Coastal' :         np.eye(11)[1],
    'Dense Forest' :    np.eye(11)[2],
    'Desert' :          np.eye(11)[3],
    'Farms' :           np.eye(11)[4],
    'Grass Land' :      np.eye(11)[5],
    'Hills' :           np.eye(11)[6],
    'Lakes' :           np.eye(11)[7],
    'River' :           np.eye(11)[8],
    'Snow' :            np.eye(11)[9],
    'Sparse Forest' :   np.eye(11)[10],
}

def one_hot_encode_geography_type(geography_type: str) -> np.ndarray:
    if geography_type == 'N,A':
        return {f'geography_type_{i}':x for i, x in enumerate(_geography_types_representations['N,A'])}
    
    representation = np.zeros(11)
    for part in geography_type.split(','):
        representation += _geography_types_representations[part]
    return representation


_urban_types_representations = {
    'N,A': np.zeros(5),
    'Dense Urban' : np.array([1,0,0,0,0]),
    'Industrial' : np.array([0,1,0,0,0]),
    'Rural' : np.array([0,0,1,0,0]),
    'Sparse Urban' : np.array([0,0,0,1,0]),
    'Urban Slum' : np.array([0,0,0,0,1]),
}

def one_hot_encode_urban_type(urban_type):
    if urban_type == 'N,A':
        return _urban_types_representations['N,A']
    
    representation = np.zeros(5)
    for part in urban_type.split(','):
        representation += _urban_types_representations[part]
    return representation

_geography_change_status = {
        'Greenland' : np.eye(10)[0],
        'Prior Construction' : np.eye(10)[1],
        'Land Cleared' : np.eye(10)[2],
        'Construction Midway' : np.eye(10)[3],
        'Construction Started' : np.eye(10)[4],
        'Materials Dumped' : np.eye(10)[5],
        'Materials Introduced' : np.eye(10)[6],
        'Operational' : np.eye(10)[7],
        'Construction Done' : np.eye(10)[8],
        'Excavation' : np.eye(10)[9],
}

def one_hot_encode_change_status(change_status: str) -> np.ndarray:
    return _geography_change_status[change_status]

