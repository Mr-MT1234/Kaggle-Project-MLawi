import random
import pandas as pd
from .models.model import Model, ChangeTypes
from dataclasses import dataclass
import numpy as np
import pickle

@dataclass
class ValidationReport:
    avg_accuracy : float
    avg_f1_scores : np.ndarray
    avg_percisions : np.ndarray
    avg_recalls : np.ndarray

    f1_scores : np.ndarray
    accuracies : np.ndarray
    percisions : np.ndarray
    recalls : np.ndarray
    confusion_matrices: np.ndarray

class Validator:
    def __init__(self, dataset: pd.DataFrame, labels: pd.DataFrame, fold_count : int =10, folds =None):
        self.dataset = dataset
        self.labels = labels

        if folds is None:
            m = len(dataset)
            fold_size = m / fold_count
            indeces = list(range(m))
            random.shuffle(indeces)
            self.folds = [ indeces[int(fold_size*i) : int(fold_size*(i+1)) ] for i in range(fold_count)]

        else:
            self.folds = folds

    def validate(self, model: Model) -> ValidationReport:
        fold_count = len(self.folds)
        
        percisions = np.empty((fold_count, len(ChangeTypes)))
        recalls = np.empty((fold_count, len(ChangeTypes)))
        accuracies = np.empty(fold_count)
        confusion_matrices = np.zeros((fold_count, len(ChangeTypes), len(ChangeTypes)))

        for fold_id, fold in enumerate(self.folds):
            training_subset = self.dataset.drop(index=fold)
            validation_subset = self.dataset.iloc[fold]
            validation_labels = self.labels.iloc[fold]

            training_subset = model.preprocess_dataset(training_subset)
            validation_subset = model.preprocess_dataset(validation_subset)

            model.train(training_subset)
            predictions = model.predict(validation_subset)

            confusion_data = validation_labels.join(predictions, lsuffix='_true', rsuffix='_predicted') \
                .groupby(['change_type_true', 'change_type_predicted']) \
                .size()
            
            for (row_true, row_predicted), count in confusion_data.items():
                confusion_matrices[fold_id, row_true, row_predicted] = count
            
            for change_type in ChangeTypes:
                change_type = change_type.value
                tp = confusion_matrices[fold_id, change_type, change_type]
                fp = confusion_matrices[fold_id, :, change_type].sum() - tp
                fn = confusion_matrices[fold_id, change_type, : ] - tp
                percisions[fold_id, change_type] = tp / (tp + fp)
                recalls[fold_id, change_type] = tp / (tp + fn)
            
            accuracies[fold_id] = np.diag(confusion_matrices[fold_id]).sum() / confusion_matrices[fold_id].sum()

        f1_scores = 2*percisions*recalls / (percisions + recalls)

        avg_accuracy = accuracies.mean()
        avg_f1_scores =  f1_scores.mean(axis=0)
        avg_percisions = percisions.mean(axis=0)
        avg_recalls = recalls.mean(axis=0)

        return ValidationReport(
            avg_accuracy,
            avg_f1_scores,
            avg_percisions,
            avg_recalls,
            f1_scores,
            accuracies,
            percisions,
            recalls,
            confusion_matrices,
        )


    def save(self, path):
        with open(path, 'wb') as file:
            data = {
                'dataset': self.dataset,
                'labels' : self.labels,
                'folds': self.folds
            }

            pickle.dump(data, file)
        

    @classmethod
    def load(cls, path) -> 'Validator':
        with open(path, 'wb') as file:
            data = pickle.load(file)
            
            dataset = data['dataset']
            labels = data['labels']
            folds = data['folds']

            validator = Validator(dataset, labels, len(folds), folds)

        return validator
