import random
import pandas as pd
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
    def __init__(self, dataset, labels, weights = None, fold_count : int=10, folds =None):
        self.dataset = dataset
        self.labels = labels
        self.weights = weights
        self.unique_labels = set(labels)

        if weights is None:
            self.weights = np.ones(len(dataset))

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

    def validate(self, model_constructor, *model_args, **model_kargs) -> ValidationReport:
        fold_count = len(self.folds)
        
        percisions = np.empty((fold_count, len(self.unique_labels)))
        recalls = np.empty((fold_count, len(self.unique_labels)))
        accuracies = np.empty(fold_count)
        confusion_matrices = np.zeros((fold_count, len(self.unique_labels), len(self.unique_labels)))

        for fold_id in range(fold_count):
            fold = self.folds[fold_id]
            model = model_constructor(*model_args, **model_kargs)

            print(f"Fold {fold_id + 1}/{fold_count}: preparing_data", end="")
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
            
            for change_type in range(len(self.unique_labels)):
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


    def save_folds(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self.folds, file)
        

    def load_folds(self, path):
        with open(path, 'wb') as file:
            folds =  pickle.load(file)
            self.folds = folds

