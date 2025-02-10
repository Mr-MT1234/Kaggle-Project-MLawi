d_prime = 60 
num_epochs = 3



import json
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.linalg as linalg
from datetime import datetime
from utils import *
from sklearn.decomposition import PCA

################# preparing the data ########################

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


# Clean dataset + sorting + some transformations

datetablea = dga.apply(clean_dataset, axis=1, result_type='expand')

dfina=datetablea.apply(transform_dataset_linear_regression, axis=1, result_type='expand')

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

# final X

Xa=np.array(mmma)

# preparing y

out={'Demolition': 0,
 'Road': 1,
 'Residential': 2,
 'Commercial': 3,
 'Industrial': 4,
 'Mega Projects': 5 }

ya=np.array(dga["change_type"])
for h in range(len(ya)):
    ya[h]=out[ya[h]]

print("length of y is:",len(ya),"length of X is:", len(Xa))


########################## X_new = X after dim reduction ############################################

#d_prime = 60 
pca = PCA(n_components=d_prime)

# Fit and transform data
X_new = pca.fit_transform(Xa)

print("the shape of X_new is : ",len(X_new), len(X_new[1]))


########################## A neural network model  ##############################################################

# Set seed for reproducibility
seed = 42  # Choose any fixed number
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # Ensures full reproducibility    

# Define the neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size=100, num_classes=6):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # First hidden layer
        self.fc2 = nn.Linear(128, 90)
        self.fc3 = nn.Linear(90, 64)  
        self.fc4 = nn.Linear(64, 32) 
        self.fc5 = nn.Linear(32, num_classes)  # Output layer
        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)  # No activation here (CrossEntropyLoss includes softmax)
        return x

# Instantiate model
input_size = d_prime
num_classes = 6
model = NeuralNet(input_size, num_classes)





# Separate samples by class
X_1 = X_new[ya == 1]  # All samples with label 1
X_0 = X_new[ya == 0]  # All samples with label 0
X_2 = X_new[ya == 2]  # All samples with label 2
X_3 = X_new[ya == 3]  # All samples with label 3
X_4 = X_new[ya == 4]  # All samples with label 4
X_5 = X_new[ya == 5]  # All samples with label 5

y_1 = ya[ya == 1]  # Labels for class 1
y_0 = ya[ya == 0]  # Labels for class 0
y_2 = ya[ya == 2]  # Labels for class 2
y_3 = ya[ya == 3]  # Labels for class 3
y_4 = ya[ya == 4]  # Labels for class 4
y_5 = ya[ya == 5]  # Labels for class 5


# Split classes separately
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size=0.15, random_state=42)
X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(X_0, y_0, test_size=0.15, random_state=42)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.15, random_state=42)
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X_3, y_3, test_size=0.15, random_state=42)
X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split(X_4, y_4, test_size=0.15, random_state=42)
X_train_5, X_test_5, y_train_5, y_test_5 = train_test_split(X_5, y_5, test_size=0.15, random_state=42)

# Combine back the training and testing sets
X_train = np.concatenate([X_train_0, X_train_1, X_train_2, X_train_3, X_train_4, X_train_5], axis=0)
y_train = np.concatenate([y_train_0, y_train_1, y_train_2, y_train_3, y_train_4, y_train_5], axis=0)

X_test = np.concatenate([X_test_0, X_test_1, X_test_2, X_test_3,X_test_4, X_test_5], axis=0)
y_test = np.concatenate([y_test_0, y_test_1, y_test_2, y_test_3, y_test_4, y_test_5], axis=0)

# Shuffle the training and test sets
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train, random_state=42)
X_test, y_test = shuffle(X_test, y_test, random_state=42)

# Print final class distribution in training set
unique, counts = np.unique(y_train, return_counts=True)
print("Training set distribution:", dict(zip(unique, counts)))


y_train = y_train.astype(int)  # Convert labels to integers
y_test = y_test.astype(int)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Classification loss
optimizer = optim.Adam(model.parameters(), lr=0.001)


################################## training the model ######################################

print(" starting the training ...")
#num_epochs = 15

for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()  # Reset gradients
        outputs = model(batch_X)  # Forward pass
        loss = criterion(outputs, batch_y)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


################################## evaluating the model ######################################




def evaluate_model(model, data_loader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for batch_X, batch_y in data_loader:
            outputs = model(batch_X)  # Forward pass
            _, predicted = torch.max(outputs, 1)  # Get class with highest probability
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()  # Count correct predictions
    
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    return accuracy

evaluate_model(model, test_loader)