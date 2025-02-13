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
    def __init__(self, dataset, labels, fold_count : int=10, folds =None):
        self.dataset = dataset
        self.labels = labels
        all_labels = set(labels)

        indeces_per_label = { label : [i for i in range(len(dataset)) if labels[i] == label] for label in all_labels }

        if folds is None:

            folds_per_label = {}

            for label, indeces in indeces_per_label.items():
                m = len(indeces)
                fold_size = m / fold_count
                random.shuffle(indeces)
                folds_per_label[label] = [ indeces[int(fold_size*i) : int(fold_size*(i+1)) ] for i in range(fold_count)]

            def concat(lists):
                out = []
                for l in lists:
                    out += l
                return out

            self.folds = [concat(folds_per_label[label][i] for label in all_labels ) for i in range(fold_count)]

        else:
            self.folds = folds

    def validate(self, models: list[Model]) -> ValidationReport:
        fold_count = len(self.folds)
        assert len(models) == fold_count, "Number of models does no match the fold_count"
        
        percisions = np.empty((fold_count, len(ChangeTypes)))
        recalls = np.empty((fold_count, len(ChangeTypes)))
        accuracies = np.empty(fold_count)
        confusion_matrices = np.zeros((fold_count, len(ChangeTypes), len(ChangeTypes)))

        for fold_id in range(fold_count):
            fold = self.folds[fold_id]
            model = models[fold_id]

            print(f"Fold {fold_id}/{fold_count}: preparing_data", end="")
            training_indeces = np.ones(len(self.dataset), dtype=bool)
            training_indeces[fold] = False
            training_subset = self.dataset[training_indeces]
            training_labels = self.labels[training_indeces]
            validation_subset = self.dataset[fold]
            validation_labels = self.labels[fold]

            print(f", training", end="")
            model.train(training_subset, training_labels)

            print(f", validating", end="")
            predictions = model.predict(validation_subset)
            

            for row_true, row_predicted in zip(validation_labels, predictions):
                confusion_matrices[fold_id, row_true, row_predicted] +=1
            
            for change_type in ChangeTypes:
                change_type = change_type.value
                tp = confusion_matrices[fold_id, change_type, change_type]
                fp = confusion_matrices[fold_id, :, change_type].sum() - tp
                fn = confusion_matrices[fold_id, change_type, : ].sum() - tp
                percisions[fold_id, change_type] = tp / (tp + fp + 0.0000001)
                recalls[fold_id, change_type] = tp / (tp + fn + 0.0000001)
            
            accuracies[fold_id] = np.diag(confusion_matrices[fold_id]).sum() / len(validation_subset)

            print(", done")

        f1_scores = 2*percisions*recalls / (percisions + recalls + 0.0000001)

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
