import json
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.linalg as linalg


#loading the data

with open("train.geojson","r") as file:
    geodata=json.load(file)

features=geodata["features"]

data1=[feature["properties"]for feature in features]

for i in range(len(data1)):
    data1[i]["geometry"]=features[i]["geometry"]["coordinates"][0]


df1a=pd.DataFrame(data1)
dga=df1a[['urban_type', 'geography_type', 'change_type','date0',
       'change_status_date0', 'date1', 'change_status_date1', 'date2',
       'change_status_date2', 'date3', 'change_status_date3', 'date4',
       'change_status_date4','img_red_mean_date1',
       'img_green_mean_date1', 'img_blue_mean_date1', 'img_red_std_date1',
       'img_green_std_date1', 'img_blue_std_date1', 'img_red_mean_date2',
       'img_green_mean_date2', 'img_blue_mean_date2', 'img_red_std_date2',
       'img_green_std_date2', 'img_blue_std_date2', 'img_red_mean_date3',
       'img_green_mean_date3', 'img_blue_mean_date3', 'img_red_std_date3',
       'img_green_std_date3', 'img_blue_std_date3', 'img_red_mean_date4',
       'img_green_mean_date4', 'img_blue_mean_date4', 'img_red_std_date4',
       'img_green_std_date4', 'img_blue_std_date4', 'img_red_mean_date5',
       'img_green_mean_date5', 'img_blue_mean_date5', 'img_red_std_date5',
       'img_green_std_date5', 'img_blue_std_date5','geometry']]

# Clean dataset
for c in dga.columns:
    dga = dga.drop(dga.index[dga[c].isnull()])



from datetime import datetime

def sort_row(row):
    to_date_time = lambda x: datetime.strptime(x, '%d-%m-%Y')

    if not row.date0 or not row.date1 or not row.date2 or not row.date3 or not row.date4:
        return {
        'date0' : None,
        'change_status_date0' : None,
        'date1' : None,
        'change_status_date1' : None,
        'date2' : None,
        'change_status_date2' : None,
        'date3' : None,
        'change_status_date3' : None,
        'date4' : None,
        'change_status_date4' : None,
        'urban_type':None,
        'geography_type':None,
        'img_red_mean_date1':None,
       'img_green_mean_date1':None,
        'img_blue_mean_date1':None,
        'img_red_std_date1':None,
       'img_green_std_date1':None,
        'img_blue_std_date1':None,
        'img_red_mean_date2':None,
       'img_green_mean_date2':None,
         'img_blue_mean_date2':None, 
         'img_red_std_date2':None,
       'img_green_std_date2':None,
         'img_blue_std_date2':None,
     'img_red_mean_date3':None,
       'img_green_mean_date3':None, 
       'img_blue_mean_date3':None, 
       'img_red_std_date3':None,
       'img_green_std_date3':None,
         'img_blue_std_date3':None,
           'img_red_mean_date4':None,
       'img_green_mean_date4':None,
         'img_blue_mean_date4':None,
           'img_red_std_date4':None,
       'img_green_std_date4':None,
         'img_blue_std_date4':None, 
         'img_red_mean_date5':None,
       'img_green_mean_date5':None,
         'img_blue_mean_date5':None,
           'img_red_std_date5':None,
       'img_green_std_date5':None, 
       'img_blue_std_date5':None,
       'geometry':None,
        }

    
    date0 = to_date_time(row.date0), row.change_status_date0 , row.img_red_mean_date1, row.img_green_mean_date1, row.img_blue_mean_date1, row.img_red_std_date1,row.img_green_std_date1, row.img_blue_std_date1
    date1 = to_date_time(row.date1), row.change_status_date1 , row.img_red_mean_date2, row.img_green_mean_date2, row.img_blue_mean_date2, row.img_red_std_date2,row.img_green_std_date2, row.img_blue_std_date2
    date2 = to_date_time(row.date2), row.change_status_date2 , row.img_red_mean_date3, row.img_green_mean_date3, row.img_blue_mean_date3, row.img_red_std_date3,row.img_green_std_date3,row.img_blue_std_date3
    date3 = to_date_time(row.date3), row.change_status_date3 , row.img_red_mean_date4, row.img_green_mean_date4, row.img_blue_mean_date4, row.img_red_std_date4,row.img_green_std_date4,row.img_blue_std_date4
    date4 = to_date_time(row.date4), row.change_status_date4 , row.img_red_mean_date5, row.img_green_mean_date5, row.img_blue_mean_date5, row.img_red_std_date5,row.img_green_std_date5,row.img_blue_std_date5

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
        'urban_type':row.urban_type,
        'geography_type':row.geography_type,
        'img_red_mean_date1':s[0][2],
       'img_green_mean_date1':s[0][3],
        'img_blue_mean_date1':s[0][4],
        'img_red_std_date1':s[0][5],
       'img_green_std_date1':s[0][6],
        'img_blue_std_date1':s[0][7],
        'img_red_mean_date2':s[1][2],
       'img_green_mean_date2':s[1][3],
        'img_blue_mean_date2':s[1][4],
        'img_red_std_date2':s[1][5],
       'img_green_std_date2':s[1][6],
        'img_blue_std_date2':s[1][7],
        'img_red_mean_date3':s[2][2],
       'img_green_mean_date3':s[2][3],
        'img_blue_mean_date3':s[2][4],
        'img_red_std_date3':s[2][5],
       'img_green_std_date3':s[2][6],
        'img_blue_std_date3':s[2][7],
        'img_red_mean_date4':s[3][2],
       'img_green_mean_date4':s[3][3],
        'img_blue_mean_date4':s[3][4],
        'img_red_std_date4':s[3][5],
       'img_green_std_date4':s[3][6],
        'img_blue_std_date4':s[3][7],
        'img_red_mean_date5':s[4][2],
       'img_green_mean_date5':s[4][3],
        'img_blue_mean_date5':s[4][4],
        'img_red_std_date5':s[4][5],
       'img_green_std_date5':s[4][6],
        'img_blue_std_date5':s[4][7],
        'geometry':row.geometry,
        
    }

datetablea = dga.apply(sort_row, axis=1, result_type='expand')


def polygon_area_perimeter(coords):
    coords = np.array(coords)
    
    # Ensure the polygon is closed (first point = last point)
    if not np.array_equal(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])  # Append first point to close polygon

    # Shoelace formula for area
    x = coords[:, 0]
    y = coords[:, 1]
    area = 0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]))

    # Compute perimeter (sum of distances between consecutive points)
    perimeter = np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
    
    return area, perimeter



def transforma(row):
    if not row.date0 or not row.date1 or not row.date2 or not row.date3 or not row.date4:
        return {
            'diff0' : None,
        'diff1' : None,
        'diff2' : None,
        'diff3' : None,
        'change_status_date0' : None,
        'change_status_date1' : None,
        'change_status_date2' : None,
        'change_status_date3' : None,
        'change_status_date4' : None,
        'urban_type':None,
        'geography_type':None,
        'img_red_mean_date1':None,
       'img_green_mean_date1':None,
        'img_blue_mean_date1':None,
        'img_red_std_date1':None,
       'img_green_std_date1':None,
        'img_blue_std_date1':None,
        'img_red_mean_date2':None,
       'img_green_mean_date2':None,
         'img_blue_mean_date2':None, 
         'img_red_std_date2':None,
       'img_green_std_date2':None,
         'img_blue_std_date2':None,
     'img_red_mean_date3':None,
       'img_green_mean_date3':None, 
       'img_blue_mean_date3':None, 
       'img_red_std_date3':None,
       'img_green_std_date3':None,
         'img_blue_std_date3':None,
           'img_red_mean_date4':None,
       'img_green_mean_date4':None,
         'img_blue_mean_date4':None,
           'img_red_std_date4':None,
       'img_green_std_date4':None,
         'img_blue_std_date4':None, 
         'img_red_mean_date5':None,
       'img_green_mean_date5':None,
         'img_blue_mean_date5':None,
           'img_red_std_date5':None,
       'img_green_std_date5':None, 
       'img_blue_std_date5':None,
       
       'area': None,
       'perimeter': None,
       'areaTperimeter': None,
        'areaSURperimeter':None,
        'compactness':None,
            
        }
    d0=pd.to_datetime(row.date0, format='%d-%m-%Y')
    d1=pd.to_datetime(row.date1, format='%d-%m-%Y')
    d2=pd.to_datetime(row.date2, format='%d-%m-%Y')
    d3=pd.to_datetime(row.date3, format='%d-%m-%Y')
    d4=pd.to_datetime(row.date4, format='%d-%m-%Y')

    a, p=polygon_area_perimeter(row.geometry)


    return {
        'diff0' : (d1-d0).days/365.25,
        'diff1' : (d2-d1).days/365.25,
        'diff2' : (d3-d2).days/365.25,
        'diff3' : (d4-d3).days/365.25,
        'change_status_date0' : row.change_status_date0,
        'change_status_date1' : row.change_status_date1,
        'change_status_date2' : row.change_status_date2,
        'change_status_date3' : row.change_status_date3,
        'change_status_date4' : row.change_status_date4,
        'urban_type':row.urban_type,
        'geography_type':row.geography_type,
                'img_red_mean_date1':row.img_red_mean_date1,
       'img_green_mean_date1':row.img_green_mean_date1,
        'img_blue_mean_date1':row.img_blue_mean_date1,
        'img_red_std_date1':row.img_red_std_date1,
       'img_green_std_date1':row.img_green_std_date1,
        'img_blue_std_date1':row.img_blue_std_date1,
        'img_red_mean_date2':row.img_red_mean_date2,
       'img_green_mean_date2':row.img_green_mean_date2,
        'img_blue_mean_date2':row.img_blue_mean_date2,
        'img_red_std_date2':row.img_red_std_date2,
       'img_green_std_date2':row.img_green_std_date2,
        'img_blue_std_date2':row.img_blue_std_date2,
        'img_red_mean_date3':row.img_red_mean_date3,
       'img_green_mean_date3':row.img_green_mean_date3,
        'img_blue_mean_date3':row.img_blue_mean_date3,
        'img_red_std_date3':row.img_red_std_date3,
       'img_green_std_date3':row.img_green_std_date3,
        'img_blue_std_date3':row.img_blue_std_date3,
        'img_red_mean_date4':row.img_red_mean_date4,
       'img_green_mean_date4':row.img_green_mean_date4,
        'img_blue_mean_date4':row.img_blue_mean_date4,
        'img_red_std_date4':row.img_red_std_date4,
       'img_green_std_date4':row.img_green_std_date4,
        'img_blue_std_date4':row.img_blue_std_date4,
        'img_red_mean_date5':row.img_red_mean_date5,
       'img_green_mean_date5':row.img_green_mean_date5,
        'img_blue_mean_date5':row.img_blue_mean_date5,
        'img_red_std_date5':row.img_red_std_date5,
       'img_green_std_date5':row.img_green_std_date5,
        'img_blue_std_date5':row.img_blue_std_date5,
        
        'area': a,
       'perimeter': p,
       'areaTperimeter': a*p,
       'areaSURperimeter':a/p,
       'compactness':p*p/(a+0.00001),

    }

dfina=datetablea.apply(transforma, axis=1, result_type='expand')

Ma=dfina.to_numpy()

catt=['Construction Done','Materials Introduced', 'Excavation', 'Land Cleared', 'Operational', 'Materials Dumped', 'Construction Midway', 'Greenland', 'Prior Construction','Construction Started']



def vectorich(elm):
    return [(catt[qq] in elm)*1 for qq in range(len(catt))]



from copy import deepcopy

mma=deepcopy(Ma)



#geography_type
geoo=['Desert', 'Sparse Forest', 'Lakes', 'Dense Forest', 'River',  'Coastal', 'Farms', 'Hills', 'Snow', 'Barren Land', 'Grass Land']

def vectorig(elm):
    return [(geoo[qq] in elm)*1 for qq in range(len(geoo))]


#urban_type
def vectorizu(elm):
    return [('Sparse Urban' in elm)*1, ('Industrial'in elm)*1, ('Rural'in elm)*1, ('Dense Urban'in elm)*1, ('Urban Slum'in elm)*1]


mmma=[[0]*55 for jjj in range(len(mma))]
for i in range(len(mma)):
    mmma[i]=list(mma[i][0:4])+ vectorich(mma[i][4])+ vectorich(mma[i][5])+ vectorich(mma[i][6])+ vectorich(mma[i][7])+ vectorich(mma[i][8])+ vectorizu(mma[i][9])+ vectorig(mma[i][10]) + list(mma[i][11:])


print(mma[3])
print(mmma[3])



Xa=np.array(mmma)



out={'Demolition': 0,
 'Road': 1,
 'Residential': 2,
 'Commercial': 3,
 'Industrial': 4,
 'Mega Projects': 5 }

ya=np.array(dga["change_type"])
for h in range(len(ya)):
    ya[h]=out[ya[h]]

ya, len(ya), len(Xa)



# Shuffle the training and test sets
from sklearn.utils import shuffle
X_train, y_train = shuffle(Xa, ya, random_state=42)

# Print final class distribution in training set
unique, counts = np.unique(y_train, return_counts=True)
print("Training set distribution:", dict(zip(unique, counts)))

y_train = y_train.astype(int)

################################################# we have now X_train and y_train ##########################################################################3

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
import numpy as np


from sklearn.ensemble import RandomForestClassifier

##################################################   the model #########################################################################################

# Generate predictions


rf_model = RandomForestClassifier(n_estimators=100, verbose=2,random_state=42)



print("rf")
rf_model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score


################################################## y_pred ######################################################################################

y_pred = rf_model.predict(X_train)
accuracy = accuracy_score(y_train, y_pred)
print(f"Accuracy: {accuracy:.4f}")