from model import Model
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from utils import *


class LinearRegression(Model):
    def __init__(self):
        self.model=LinearRegression()

    def preprocess_dataset(self, dataset : pd.DataFrame):
        # Removing rows that contain None + sorting the dataset
        clean_dataset = clean_dataset(dataset)
        # deriving some features from geometry and dates
        clean_dataset=transform_dataset_linear_regression(clean_dataset)

        # One-hot encoding
        clean_dataset = pd.get_dummies(clean_dataset, columns=['change_status_date0', 'change_status_date1', 'change_status_date2', 'change_status_date3', 'change_status_date4'], dtype='uint64')
        for col in ['geography_type', 'urban_type']:
            one_hot = clean_dataset[col].str.get_dummies(sep=',')
            clean_dataset = pd.concat([clean_dataset, one_hot], axis=1)
            clean_dataset.drop(columns=[col], inplace=True)


        return clean_dataset
        

    def train(self, dataset):
        # Separate samples and labels
        X_train, y_train = dataset.drop('change_type', axis=1), dataset['change_type']

        self.model.fit(X_train,y_train)
    
    def predict(self, input):
        y_pred = self.model.predict(input)
        return y_pred
    
    def evaluate(self, y_test, y_pred):
        assert len(y_test)==len(y_pred)

        tp=len([1 for k in range(len(y_test)) if y_test[k]==y_pred[k]])
        a = tp/len(y_test)

        mse = mean_squared_error(y_test, y_pred)

        print("Accuracy:", a)
        print("Mean Squared Error (MSE):", mse)