from matplotlib import pyplot as plt
import numpy as np
# matplotlib.use('TkAgg')
# from hungarian import hungarian_algorithm_method
import torch
# from torch_geometric.nn import GCNConv
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, precision_score, recall_score

import torch_geometric as G
from sklearn.metrics import confusion_matrix

import seaborn as sns

# from torch_geometric.datasets import Planetoid
# from torch_geometric.transforms import NormalizeFeatures
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float)
torch.autograd.set_detect_anomaly(True)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
import warnings
warnings.filterwarnings('always')
ORIG_TIME_CHANNELS = 128
SPACE_CHANNELS = 64

TIME_CHANNELS = ORIG_TIME_CHANNELS // 2

TRACKED_METRICS = [
    ("Accuracy", accuracy_score),
    ("Precision", lambda x, y: precision_score(x, y,  average='macro', zero_division= 0.0)),
    ("Recall", lambda x, y: recall_score(x, y,  average='macro', zero_division= 0.0))
]


def _convert_output(model_out, as_one_hot=False, as_numpy=True):
    with torch.no_grad():
        pred_classes = torch.argmax(model_out, 1)
        if as_one_hot:
            out = torch.zeros_like(model_out, dtype=torch.int)
            for p in range(pred_classes.shape[0]):
                out[p, pred_classes[p]] = 1

        else:
            out = pred_classes

        if as_numpy:
            out = np.array(out)
        return out

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


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, adj, dropout=0.2):
        super(GraphConv, self).__init__()
        self.graph_attn = nn.Parameter(adj)
        nn.init.constant_(self.graph_attn, 1)
        self.A = adj#torch.tensor(adj, requires_grad=False)

        # Create Convolutions for each neighbourhood
        self.num_subset = adj.shape[0]
        self.g_convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.g_convs.append(
                nn.Conv2d(in_channels, out_channels, 1)
            )

        # Create Residual Connections
        if in_channels != out_channels:
            self.gcn_residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            # Do nothing
            self.gcn_residual = lambda x: x

        # Create BatchNorm layers and dropout
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, X):
        batches, channels, times, nodes = X.size()
        A = self.A
        A = A * self.graph_attn # Apply attention
        hidden_vals = None

        # Convolve for each subset
        for i in range(self.num_subset):
            x_a = X.view(batches, times * channels, nodes)

            # mul_res = torch.matmul(x_a, A[i])
            # view_res = mul_res.view(batches, channels, times, nodes)

            z = self.g_convs[i](
                # Apply normalisation to nodes
                torch.matmul(x_a, A[i]).view(batches, channels, times, nodes)
            )
            hidden_vals = z if hidden_vals is None else z + hidden_vals
        
        # Apply batchnorm and residual
        out = self.batchnorm(hidden_vals)
        out = self.dropout(out)
        out += self.gcn_residual(X)
        return self.relu(out)


class TemporalGatedConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(TemporalGatedConv, self).__init__()
        # self.conv = nn.Conv1d(features, out_channels, kernel_size=3)
        # self.flatten = nn.Flatten(start_dim = 0, end_dim=1)
        # self.unflatten = nn.Unflatten(0, (-1, nodes))
        # self.glu = nn.GLU(dim=-1)
        pad = (kernel_size - 1) // 2
        self.t_conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1), padding=(pad, 0),
            stride=(stride, 1)
        )
        self.batchnorm = nn.BatchNorm2d(out_channels)
    
    def forward(self, X):
        """
        X is shape (batch, channel, time, nodes)
        """
        # TODO: Add a third dimension for certainty in dataset
        # TODO: 
        # inp = X.permute(0, 1, 3, 2) 
        # outp_orig = torch.cat([
        #     torch.unsqueeze(self.conv(inp[:, :, :, i]), 0) for i in range(inp.shape[-1])
        # ]).permute(1, 2, 0, 3)
        # print(outp_orig.shape)
        # # quit()

        # inp = X.permute(0, 2, 3, 1) # (n, nodes, features, time)
        # inp = self.flatten(inp)#torch.flatten(inp, end_dim=1)
        # outp = self.conv(inp)
        # outp = self.unflatten(outp).permute(0, 3, 1, 2)#torch.unflatten(outp, 0, (-1, self.nodes)).permute(0, 2, 1, 3)
        
        
        # outp_g = self.glu(outp)
        return self.batchnorm(
            self.t_conv(X)
        )
    





class MySTGCNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, A, stride=None, residual=True):
        
        super(MySTGCNBlock, self).__init__()
        # self.time1 = TemporalGatedConv(features, TIME_CHANNELS, nodes)
        # self.convblock = G.torch_geometric.nn.conv.GraphConv(TIME_CHANNELS//2, SPACE_CHANNELS)        
        # self.time2 = TemporalGatedConv(TIME_CHANNELS//2, TIME_CHANNELS, nodes)
        # self.flatten = nn.Flatten(start_dim=0, end_dim=1)
        # self.unflatten = nn.Unflatten(0, (-1, timesteps))
        if stride is None:
            stride = 1
        self.gcn = GraphConv(in_channels, out_channels, A)
        self.tcn = TemporalGatedConv(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()

        if not residual:
            self.res = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.res = lambda x: x

        else:
            self.res = TemporalGatedConv(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, X):
        # in_size = X.shape
        # x1 = self.time1(X)
        # unflatten_size = x1.shape[1]
        # orig_size = x1.shape
        # x1 = self.flatten(x1)#torch.flatten(x1, start_dim=0, end_dim=1)
        # flattened_size = x1.shape
        # x2 = self.convblock(x1, edge_mat)
        # # print(x2.shape, orig_size, flattened_size, in_size)
        # # quit()
        # x2 = torch.unflatten(x1, 0, (-1, unflatten_size))#self.unflatten(x1)
        # # x2 = torch.unflatten(x1, dim=0, sizes=orig_size)
        # # quit()
        # x3 = self.time2(x2)
        # return x3

        # Time(Graph) + Residual
        out = self.tcn(self.gcn(X)) + self.res(X)
        return self.relu(out)
    
class MySTGCN(nn.Module):

    def __init__(self, classes, nodes, in_channels, gso, blocks=None, strides=None):
        
        super(MySTGCN, self).__init__()
        # self.block1 = MySTGCNBlock(timesteps, nodes, features)
        # self.block2 = MySTGCNBlock(timesteps, nodes, TIME_CHANNELS//2)
        # self.time_full = nn.Linear(nodes * features, 1024)
        # self.full1 = nn.LazyLinear( 1024)
        # self.full2 = nn.Linear(1024, 1024)
        # self.FC = nn.Linear(1024, classes)
        # self.flatten = nn.Flatten(start_dim=1)
        # self.time_flatten = nn.Flatten(start_dim=0, end_dim=1)
        # self.time_unflatten = nn.Unflatten(0, (-1, 64))
        # self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(1)

        if blocks is None:
            blocks= [
                64, 
                64, 
                64, 
                64, 
                128, 
                128, 
                128, 
                256, 
                256, 
                256
            ]

        if strides is None:
            strides = [
                None,
                None,
                None,
                None,
                2,
                None,
                None,
                2,
                None,
                None
            ]

        if len(blocks) != len(strides):
            raise ValueError(f"Blocks and strides must be same length, got len(blocks) = {len(blocks)} and len(strides) = {len(strides)}")
        
        # BatchNorm Layer
        self.batchnorm = nn.BatchNorm1d(in_channels * nodes)

        # Create Layers
        layer_dict = {}
        prev_block = in_channels
        for i, (block, stride) in enumerate(zip(blocks, strides)):
            print(f'MySTGCNBlock({prev_block}, {block})')
            layer_dict[f'layer{i}'] = MySTGCNBlock(
                prev_block, block, gso, residual= i!=0, stride=stride
            )
            prev_block = block
        self.layers = nn.ModuleDict(layer_dict)

        self.class_layer = nn.Linear(prev_block, classes)
        self.softmax = nn.Softmax(1)



    def forward(self, X):
        """
        X: Shape (batch, channels, times, nodes)
        """
        # x1 = self.relu(self.block1(X, edge_mat))
        # x2 = self.relu(self.block2(x1, edge_mat))
        

        # x3 = self.flatten(x2)#torch.flatten(x2, start_dim=1)
        # x4 = self.full2(self.full1(x3))
        # x5 = self.FC(x4)
        # # print(x3)
        # return self.softmax(x5)
        X = X.permute(0, 3, 1, 2)

        B, C, T, N = X.size()
        x = X.permute(0, 3, 1, 2).contiguous().view(B, N * C, T)
        x = self.batchnorm(x)
        x = x.view(B, N, C, T).permute(0, 2, 3, 1).contiguous().view(B, C, T, N)
        for i in range(len(self.layers)):
            x = self.layers[f'layer{i}'](x)

        c_new = x.size(1)
        x = x.view(B, c_new, -1)
        x = x.mean(2)
        return self.softmax(self.class_layer(x))
    
    def predict(self, X, as_one_hot=False, as_numpy=True):
        with torch.no_grad():
            model_out = self.forward(X)
            return _convert_output(model_out, as_one_hot=as_one_hot, as_numpy=as_numpy)
        


        



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
        self.X_val = torch.tensor(dataset.X_val).float()
        self.X_test = torch.tensor(dataset.X_test).float().to(device)
        self.gso = torch.tensor(dataset.gso, dtype=torch.float)
        self.y = torch.tensor(dataset.y_train)
        self.y_val = torch.tensor(dataset.y_val)
        self.y_test = torch.tensor(dataset.y_test).to(device)
        self.n_features = dim(dataset.X_train)[-1] // num_dims

        self.n_classes = dataset.n_classes
        self.edges = torch.tensor(dataset.edge_matrix).to(device)
        # self.adj_mat = torch.tensor(self.dataset.create_graph_shift_operator()).int()



       

        num_nodes = self.X.shape[2]
        # print(num_nodes, self.X.shape)
        # quit()
        num_features = self.X.shape[-1]
        num_timesteps = self.X.shape[1]

        self.model = MySTGCN(self.n_classes, num_nodes, num_features, self.gso).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4)#, weight_decay=5e-4)
        self.criterion = torch.nn.CrossEntropyLoss()
       

    def _train(self, X_train=None, y_train=None, batch_size=None):
        if X_train is None:
            X_train = self.X
            y_train = self.y
        if batch_size is None:
            batch_size = X_train.shape[0]
        # print(y_train.shape)
        # quit()
        loss = torch.tensor(0).double().to(device)#self.criterion(out, y_train)
        out_ret = None
        for b in range(X_train.shape[0] // batch_size):
            out = self.model(X_train[b * batch_size: (b+1) * batch_size,...].to(device))
            if out_ret is None:
                out_ret = out
            else:
                out_ret = torch.stack((out_ret, out))
            loss += self.criterion(out, y_train[b * batch_size: (b+1) * batch_size].to(device))
        loss /= X_train.shape[0] // batch_size
        # for p in self.model.parameters():
        #     print(p.grad)
        # quit()
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return loss, _convert_output(out_ret, as_one_hot=True)
    
    def train(self, epochs, X_train=None, y_train=None, val_set=None, batch_size=None):
        if y_train is None:
            y_train = self.y
        for epoch in range(1, epochs + 1):
            loss, out = self._train(batch_size=batch_size)
            
            val_message = " || Train: "
            val_metrics = [f"{x}: {y(y_train, out):.3f}" for x, y in TRACKED_METRICS]
            val_message += " | ".join(val_metrics)
            if val_set is not None:
                X_val, y_val = val_set
                out = self.model.predict(X_val, as_one_hot=True)
                val_message += " || Validation: "
                val_metrics = [f"{x}: {y(y_val, out):.3f}" for x, y in TRACKED_METRICS]
                val_message += " | ".join(val_metrics)

                
            print("===============================")
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}' + val_message)
            print("===============================")

    def test(self):
        self.model.eval()
        out = self.model(self.X_test, self.edges)
        pred = torch.zeros_like(out)
        pred[:, out.argmax(dim=1)] = 1
        test_val = self.y_test.argmax(dim=1)
        print(pred.shape, self.y_test.shape)
        print(out)
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
        self.train(1000, batch_size=None, val_set=(self.X_val, self.y_val))
        acc = self.test()
        print(f"Accuracy: {acc:.2f}")

    def predict(self, X):
        return self.model(X, self.edges)
    
