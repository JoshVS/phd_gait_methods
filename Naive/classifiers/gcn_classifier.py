import numpy as np
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
from scipy.optimize import linear_sum_assignment as hungarian_algorithm_method
# from hungarian import hungarian_algorithm_method
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from sklearn.manifold import TSNE

def _dim(l, check_for_error):
    if type(l) != list and type(l) != np.ndarray:
        return []
    else:
        if type(l[0]) == list and check_for_error:
            next_dim = len(l[0])
            for mini_l in l[1:]:
                if len(mini_l) != next_dim:
                    raise ValueError("Array is sparse")
        return [len(l)] + _dim(l[0], check_for_error)

def dim(l, check_for_error=False):
    return tuple(_dim(l, check_for_error))

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GCNClassifier():
    """
    Classifier must take a dataset in initialiser

    Methods included:

    train(X, y)
    generate_test_set_results()
    predict(X)

    TODO: X must be of shape (n_samples, n_keypoints, 2)
    TODO: edges must be of shape (n_edges, 2)
    """
    def __init__(self, dataset, num_dims=3, num_phases=4):
        self.dataset = dataset
        self.X = torch.tensor(dataset.X_train)[:,0,:]
        
        self.y = torch.tensor(dataset.y_train)
        self.n_features = dim(dataset.X_train)[-1] // num_dims

        self.n_classes = dataset.n_classes
        self.edges = torch.tensor(dataset.edge_matrix)
        # print(self.X.shape)
        # print(self.edges.shape)
        # quit()

        self.classifier = GCN(self.n_features, self.n_classes, 16)
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.01, weight_decay=5e-4)
        self.criterion = torch.nn.CrossEntropyLoss()

    def _train(self, X_train=None, y_train=None):
        if X_train is None:
            X_train = self.X
            y_train = self.y
        self.classifier.train()
        self.optimizer.zero_grad()
        out = self.classifier(X_train, self.edges)
        loss = self.criterion(out, y_train)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def train(self, epochs, X_train=None, y_train=None):
        for epoch in range(1, epochs + 1):
            loss = self._train()
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    def test(self):
        self.model.eval()
        out = self.model(self.dataset.X_test, self.edges)
        pred = out.argmax(dim=1)
        test_correct = pred == self.dataset.y_test
        test_acc = int(test_correct.sum()) / len(self.dataset.y_train)
        return test_acc
    
    def generate_test_set_results(self):
        self.train(100)
        acc = self.test()
        print(f"Accuracy: {acc:.2f}")

    def predict(self, X):
        return self.classifier(X, self.edges)