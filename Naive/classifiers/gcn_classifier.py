from matplotlib import pyplot as plt
import numpy as np
# matplotlib.use('TkAgg')
# from hungarian import hungarian_algorithm_method
import torch
# from torch_geometric.nn import GCNConv
import torch.nn.functional as F

import torch_geometric as G
from sklearn.metrics import confusion_matrix

import seaborn as sns

# from torch_geometric.datasets import Planetoid
# from torch_geometric.transforms import NormalizeFeatures
import math
import torch
import torch.nn as nn
import torch.nn.functional as F



def to_one_hot(y):
    n_c = len(np.unique(y))

    to_out = np.zeros((len(y), n_c))
    for i in range(to_out.shape[0]):
        to_out[i, y[i]] = 1
    return to_out

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

class TemporalGatedConv(nn.Module):

    def __init__(self, time_channels):

        
        super(TemporalGatedConv, self).__init__()
        self.conv = nn.Conv1d(time_channels, 64, kernel_size=1)
    
    def forward(self, X):
        """
        X is shape (n, time, nodes, features)
        """
        inp = X.permute(0, 1, 3, 2) 
        outp = torch.cat([
            torch.unsqueeze(self.conv(inp[:, :, :, i]), 0) for i in range(inp.shape[-1])
        ]).permute(1, 2, 0, 3)
        
        
        outp_g = F.glu(outp, dim=1)
        return outp_g
    





class MySTGCNBlock(nn.Module):

    def __init__(self, time_channels):
        
        super(MySTGCNBlock, self).__init__()
        self.time1 = TemporalGatedConv(time_channels)
        self.convblock = G.torch_geometric.nn.conv.GraphConv(2, 16)        
        self.time2 = TemporalGatedConv(32)

    def forward(self, X, edge_mat):
        x1 = self.time1(X)
        orig_size = x1.shape[:2]
        x1 = torch.flatten(x1, start_dim=0, end_dim=1)
        # print(x1.shape)
        # quit()
        x2 = self.convblock(x1, edge_mat)
        x2 = torch.unflatten(x1, dim=0, sizes=orig_size)
        x3 = self.time2(x2)
        return x3
    
class MySTGCN(nn.Module):

    def __init__(self, time_channels, nodes, features, classes):
        
        super(MySTGCN, self).__init__()
        self.block1 = MySTGCNBlock(time_channels)
        self.block2 = MySTGCNBlock(32)
        self.FC = nn.Linear(32 * nodes * features, classes)

    def forward(self, X, edge_mat):
        x1 = self.block1(X, edge_mat)
        x2 = self.block2(x1, edge_mat)
        x3 = torch.flatten(x2, start_dim=1)
        x3 = self.FC(x3)
        return F.softmax(x3)

class GCNClassifier():
    """
    Classifier must take a dataset in initialiser

    Methods included:

    train(X, y)
    generate_test_set_results()
    predict(X)

    """
    def __init__(self, dataset, num_dims=3, num_phases=4):
        self.dataset = dataset
        self.X = torch.tensor(dataset.X_train).float()
        self.X_test = torch.tensor(dataset.X_test).float()
        self.y = torch.tensor(to_one_hot(dataset.y_train))
        self.y_test = torch.tensor(to_one_hot(dataset.y_test))
        self.n_features = dim(dataset.X_train)[-1] // num_dims

        self.n_classes = dataset.n_classes
        self.edges = torch.tensor(dataset.edge_matrix)
        self.adj_mat = torch.tensor(self.dataset.create_graph_shift_operator()).int()


        blocks = []
        blocks.append([self.X.shape[1]])
        blocks.append([64, 16, 64])
        blocks.append([64, 16, 64])
        blocks.append([self.X.shape[-1]])

       

        num_nodes = self.X.shape[2]
        # print(num_nodes, self.X.shape)
        # quit()
        num_features = self.X.shape[-1]
        num_timesteps = self.X.shape[1]

        self.model = MySTGCN(num_timesteps, num_nodes, num_features, self.n_classes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)#, weight_decay=5e-4)
        self.criterion = torch.nn.CrossEntropyLoss()
       

    def _train(self, X_train=None, y_train=None):
        if X_train is None:
            X_train = self.X
            y_train = self.y
        out = self.model(X_train, self.edges)
        loss = self.criterion(out, y_train)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss
    
    def train(self, epochs, X_train=None, y_train=None):
        for epoch in range(1, epochs + 1):
            loss = self._train()
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    def test(self):
        self.model.eval()
        out = self.model(self.X_test, self.edges)
        print(out)
        quit()
        pred = torch.zeros_like(out)
        pred[:, out.argmax(dim=1)] = 1
        test_val = self.y_test.argmax(dim=1)
        print(pred.shape, self.y_test.shape)
        print(pred)
        quit()
        # print(out)
        # quit()
        cm = confusion_matrix(test_val.detach().numpy(), out.detach().numpy())
        sns.heatmap(cm, annot=True)
        plt.show()
        test_correct = pred == test_val
        test_acc = int(test_correct.sum()) / len(self.dataset.y_train)
        return test_acc
    
    def generate_test_set_results(self):
        self.train(10)
        acc = self.test()
        print(f"Accuracy: {acc:.2f}")

    def predict(self, X):
        return self.model(X, self.edges)
    
