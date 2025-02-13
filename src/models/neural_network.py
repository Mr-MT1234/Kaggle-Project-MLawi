import torch
import torch.nn as nn
from .model import Model, ChangeTypes
import pandas as pd
from .. import utils
from torch.optim.adam import Adam
from datetime import datetime
from torch.utils.data import DataLoader
import math
import numpy as np
from  .. import utils
class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()

        hidden_layers = [ (
            nn.Linear(in_features=hidden_dims[i], out_features=hidden_dims[i+1]),
            nn.ReLU()
            ) 
            for i in range(len(hidden_dims) - 1)
        ]

        hidden_layers = [l for group in hidden_layers for l in group]

        self.layers = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dims[0]),
            nn.ReLU(),
            *hidden_layers,
            nn.Linear(in_features=hidden_dims[-1], out_features=output_dim),
            nn.Softmax(dim=-1)
        )


    def forward(self, input):
        return self.layers(input)
    

FEATURE_COUNT = 105
class NeuralNetModel(Model):
    def __init__(self, hidden_dims, lr=1e-3, training_epochs=100, batch_size=32, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"no device specified: {device} was chosen")

        self.lr = lr
        self.net = NeuralNet(FEATURE_COUNT, hidden_dims, len(ChangeTypes)).to(device)
        self.device = device
        self.optimizer = Adam(self.net.parameters(), lr)
        self.epochs = training_epochs
        self.batch_size = batch_size

        self.net.eval()

    def preprocess_dataset(self, dataset: pd.DataFrame) -> torch.Tensor:
        urban_type = pd.DataFrame(dataset.urban_type)
        geography_type = pd.DataFrame(dataset.geography_type)
        geometry = pd.DataFrame(dataset.geometry)
        change_status_data0 = pd.DataFrame(dataset.change_status_date0)
        change_status_date1 = pd.DataFrame(dataset.change_status_date1)
        change_status_date2 = pd.DataFrame(dataset.change_status_date2)
        change_status_date3 = pd.DataFrame(dataset.change_status_date3)
        change_status_date4 = pd.DataFrame(dataset.change_status_date4)
        dates = pd.DataFrame(dataset[['date0','date1','date2','date3','date4',]])

        dataset = dataset.drop(columns=['urban_type','geography_type','geometry',
                                        'change_status_date0','change_status_date1','change_status_date2','change_status_date3','change_status_date4',
                                        'date0', 'date1','date2','date3','date4'])

        def map_urban_type(row):
            representation = utils.one_hot_encode_urban_type(row.urban_type)
            return { f'urban_type{i}': component for i, component in enumerate(representation) }
        def map_geography_type(row):
            representation = utils.one_hot_encode_geography_type(row.geography_type)
            return { f'geography_type{i}': component for i, component in enumerate(representation) }
        def map_change_status(row, index=0):
            representation = utils.one_hot_encode_change_status(row[f'change_status_date{index}'])
            return { f'change_status_date{i}_{index}': component for i, component in enumerate(representation) }
        def map_geometry(row):
            geometry = row.geometry
            x,y = geometry.centroid.coords[0]
            return { 
                'geometry_area': geometry.area,
                'geometry_perimeter': geometry.length,
                'geometry_compactness': geometry.length**2 / (geometry.area +0.00001),
                'geomety_centroid_x' : x,
                'geomety_centroid_y' : y
            }
        def map_dates(row):
            dates = [row.date0 - pd.Timestamp(),row.date1, row.date2, row.date3, row.date4]

            delta1 = date1 - date0
            delta2 = date2 - date0
            delta3 = date3 - date0
            delta4 = date4 - date0
            return { 
                'date0': delta1.days,
                'date1': delta1.days,
                'date2': delta2.days,
                'date3': delta3.days,
                'date4': delta4.days,
            }


        urban_type = urban_type.apply(map_urban_type, axis=1, result_type='expand')
        geography_type = geography_type.apply(map_geography_type, axis=1, result_type="expand")
        geometry = geometry.apply(map_geometry, axis=1, result_type="expand")
        change_status_data0 = change_status_data0.apply(map_change_status, axis=1, result_type="expand", args=(0,))
        change_status_date1 = change_status_date1.apply(map_change_status, axis=1, result_type="expand", args=(1,))
        change_status_date2 = change_status_date2.apply(map_change_status, axis=1, result_type="expand", args=(2,))
        change_status_date3 = change_status_date3.apply(map_change_status, axis=1, result_type="expand", args=(3,))
        change_status_date4 = change_status_date4.apply(map_change_status, axis=1, result_type="expand", args=(4,))
        dates = dates.apply(map_dates, axis=1, result_type="expand")

        dataset = dataset.join(urban_type).join(geography_type).join(geometry) \
            .join(change_status_data0).join(change_status_date1).join(change_status_date2) \
            .join(change_status_date3).join(change_status_date4).join(dates).fillna(0)
        
        assert len(dataset.columns) - 1 == FEATURE_COUNT, f"Got an unexpected Feature count, may be the input to NeuralNetModel was changed withoout  \
        updating the FEATURE_COUNT constant (got {len(dataset.columns) - 1}, while FEATURE_COUNT = {FEATURE_COUNT})"

        data = torch.tensor(dataset.to_numpy())

        return data

    def train(self, dataset):
        self.net.train()
        
        loader = DataLoader(dataset, self.batch_size, shuffle=True)
        iterations = math.ceil(len(dataset) / self.batch_size)
        for epoch in range(1, self.epochs+1):
            total_loss = 0 
            for i,(data, labels) in enumerate(loader):
                data = data.to(self.device)
                labels = labels.to(self.device)
                predictions = self.net(data)

                loss = nn.functional.cross_entropy(predictions, labels)

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()
                total_loss += loss.item() 

                print(f"\rTraining {i}/{iterations} {epoch}/{self.epochs}", end='')
            print(f"\n Loss: {total_loss}")

        self.net.eval()
        
    
    def predict(self, input):
        return self.net(input.to(self.device)).argmax(axis=1)



    