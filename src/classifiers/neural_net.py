import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset
from torch.optim.adam import Adam
from .classifier import Classifier
import numpy as np
import utils

class Net(nn.Module):
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
    

FEATURE_COUNT = 105-45
class NeuralNet(Classifier):
    def __init__(self, input_dimension, hidden_dims, output_dimension, lr=1e-3, training_epochs=30, batch_size=256, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"no device specified: {device} was chosen")

        self.lr = lr
        self.net = Net(input_dimension, hidden_dims, output_dimension).to(device)
        self.device = device
        self.optimizer = Adam(self.net.parameters(), lr)
        self.epochs = training_epochs
        self.batch_size = batch_size
        self.losses = []

        self.net.eval()

    def train(self, dataset, labels, weights=None):
        if not (weights is None):
            sampler = WeightedRandomSampler(weights, len(dataset))
        else:
            sampler = None
            
        self.net.train()
        labels_one_hot = np.zeros((labels.shape[0], 6))
        labels_one_hot[range(len(labels)), labels.astype(int)] = 1

        dataset = TensorDataset(torch.tensor(dataset), torch.tensor(labels_one_hot))
        loader = DataLoader(dataset, self.batch_size, sampler=sampler)
        iterations = np.ceil(len(dataset) / self.batch_size)

        print("training ...")
        for epoch in range(1, self.epochs+1):
            total_loss = 0 
            for iteration,(data, labels) in enumerate(loader, 1):
                data = data.to(self.device)
                labels = labels.to(self.device)
                predictions = self.net(data)

                loss = nn.functional.cross_entropy(predictions, labels)

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()
                total_loss += loss.item() 

                progress_bar = utils.make_progress_bar(iteration/iterations, 100)

                print(f"\rEpoch {epoch}/{self.epochs} {progress_bar} {iteration}/{iterations}", end="")
            self.losses.append(total_loss)
            print("\n Epoch finiched with loss:", total_loss)

        self.net.eval()
    
    def predict(self, input):
        return self.net(torch.tensor(input).to(self.device)).argmax(axis=1).cpu().numpy().astype(np.int32)