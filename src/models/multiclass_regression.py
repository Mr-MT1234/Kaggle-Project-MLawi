from model import Model

class MulticlassRegression(Model):
    def __init__(self):
        super().__init__()
    
    def preprocess_dataset(self, dataset):
        ...
        
    def train(self, dataset):
        ...
    
    def predict(self, input):
        ...
