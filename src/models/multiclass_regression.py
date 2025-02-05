from model import Model
from utils import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

class MulticlassRegression(Model):
    def __init__(self):
        super().__init__()
    
    def preprocess_dataset(self, dataset):
    
        clean_dataset = dataset
        # Remove rows that contain None
        for c in clean_dataset.columns:
            clean_dataset.drop(clean_dataset.index[clean_dataset[c].isnull()], inplace=True)
            
        # Drop the column 'index'
        clean_dataset = clean_dataset.drop(columns='index')
        
        # Filter the DataFrame to exclude rows where 'geography_type' or 'urban_type' are 'N,A'
        clean_dataset = clean_dataset.loc[(clean_dataset['geography_type'] != 'N,A') & (clean_dataset['urban_type'] != 'N,A')]
        
        # Converting the dates to timestumps (in days)
        clean_dataset[['date0', 'date1', 'date2', 'date3', 'date4']] = clean_dataset[['date0', 'date1', 'date2', 'date3', 'date4']].apply(lambda x : pd.to_datetime(x, format="%d-%m-%Y").astype('int64') / (10**9*60*60*24), axis=1)

        # Only considering the area of the polygons
        clean_dataset['geometry'] = clean_dataset['geometry'].apply(lambda x : x.area)

        # One-hot encoding
        clean_dataset = pd.get_dummies(clean_dataset, columns=['change_status_date0', 'change_status_date1', 'change_status_date2', 'change_status_date3', 'change_status_date4'], dtype='uint64')
        for col in ['geography_type', 'urban_type']:
            one_hot = clean_dataset[col].str.get_dummies(sep=',')
            clean_dataset = pd.concat([clean_dataset, one_hot], axis=1)
            clean_dataset.drop(columns=[col], inplace=True)
            
        # Label encoding for the labels
        clean_dataset['target'] = pd.factorize(clean_dataset['change_type'])[0]
        print(clean_dataset.head())
        clean_dataset.drop(columns=['change_type'], inplace=True)
        
        return clean_dataset
        
    def train(self, dataset):
        
        # Separate samples and labels
        X_train, y_train = dataset.drop('target', axis=1), dataset['target']

        # Create and train logistic regression model
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
        model.fit(X_train, y_train)
        
        # Update the model
        self.model = model
        
    def predict(self, input):

        y_pred = self.model.predict(input)
        return y_pred
    
    def evaluate(self, y_test, y_pred):

        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        print("Accuracy:", accuracy)
        print("Confusion Matrix:\n", conf_matrix)
