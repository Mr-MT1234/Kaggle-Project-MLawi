from .classifier import Classifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class RandomForest(Classifier):
    def __init__(self, nb_trees, max_depth):
        self.clf = RandomForestClassifier(nb_trees, max_depth=max_depth)
    
    def train(self, data, labels, weights=None):
        self.clf.fit(data, labels, sample_weight=weights)
    
    def predict(self, input):
        return self.clf.predict(input)