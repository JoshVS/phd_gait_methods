"""
Modified based on: https://github.com/open-mmlab/mmskeleton
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from graphs.mpg import MediapipeGraph
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from torchmetrics.functional import precision, recall
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
import matplotlib.pyplot as plt
from torch.nn.functional import softmax
torch.set_default_dtype(torch.double)

device = "cuda" if torch.cuda.is_available() else "cpu"


def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

force_cudnn_initialization()
# Hi Josh, 

# Please find attached model code for the stgcn model as well as the graph creation class for mediapipe. 
# The input shape for the st-gcn model is [batch_size, channels, number_of_frames, nodes, M]
# Channels is usually 3 or 2 depending if you dealing with 3d or 2d coordinates
# M is the number of people in the clip, for me it is always 1 person pose data I'm working with.

def weights_init(module_, bs=1):
    if isinstance(module_, nn.Conv2d) and bs == 1:
        nn.init.kaiming_normal_(module_.weight, mode='fan_out')
        nn.init.constant_(module_.bias, 0)
    elif isinstance(module_, nn.Conv2d) and bs != 1:
        nn.init.normal_(module_.weight, 0,
                        math.sqrt(2. / (module_.weight.size(0) * module_.weight.size(1) * module_.weight.size(2) * bs)))
        nn.init.constant_(module_.bias, 0)
    elif isinstance(module_, nn.BatchNorm2d):
        nn.init.constant_(module_.weight, bs)
        nn.init.constant_(module_.bias, 0)
    elif isinstance(module_, nn.Linear):
        nn.init.normal_(module_.weight, 0, math.sqrt(2. / bs))


class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, A, cuda_, dropout=0.2):
        super(GraphConvolution, self).__init__()
        self.cuda_ = cuda_
        self.graph_attn = nn.Parameter(torch.from_numpy(A.astype(np.float32))) #graph_attn is the neighbourhoods - how is it represented?
        nn.init.constant_(self.graph_attn, 1)
        self.A = Variable(torch.from_numpy(A.astype(np.float64)), requires_grad=False)

        # Create Convolutions for each neighbourhood
        self.num_subset = 3 # number of neighbourhoods
        self.g_conv = nn.ModuleList()
        for i in range(self.num_subset):
            self.g_conv.append(nn.Conv2d(in_channels, out_channels, 1)) # different convolutions for each neighbourhood
            weights_init(self.g_conv[i], bs=self.num_subset)

        # Residual connections
        if in_channels != out_channels:
            self.gcn_residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
            weights_init(self.gcn_residual[0], bs=1)
            weights_init(self.gcn_residual[1], bs=1)
        else:
            self.gcn_residual = lambda x: x

        # Create batch norm layers and dropout
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        weights_init(self.bn, bs=1e-6)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x: (batch * people, channels, times, nodes)
        """
        N, C, T, V = x.size() # (batch, channels, timesteps, nodes)
        if self.cuda_:
            A = self.A.cuda(x.get_device())
        else:
            A = self.A
        A = A * self.graph_attn # apply neighbourhoods to edges
        hidden_ = None

        # Convolution for each neighbourhood
        for i in range(self.num_subset):
            x_a = x.view(N, C * T, V) #(batch, time * channel, nodes)

            # Find nodes for this neighbourhood
            # x_a: (batch, time * channel, nodes)
            # A[i]: (nodes)

            # output: (batch, channel, time, nodes)
            # Effect: applies normalisation
            z = self.g_conv[i](torch.matmul(x_a, A[i]).view(N, C, T, V))
            hidden_ = z + hidden_ if hidden_ is not None else z
        hidden_ = self.bn(hidden_)
        hidden_ = self.dropout(hidden_)
        hidden_ += self.gcn_residual(x)
        return self.relu(hidden_)


class TemporalConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(TemporalConvolution, self).__init__()

        pad = int((kernel_size - 1) / 2)
        self.t_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1),
                                padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        weights_init(self.t_conv, bs=1)
        weights_init(self.bn, bs=1)

    def forward(self, x):
        """
        X: Shape (batch, channel, time, nodes)
        """
        x = self.bn(self.t_conv(x))
        return x


class ST_GCN_block(nn.Module):
    def __init__(self, in_channels, out_channels, A, cuda_=False, stride=1, residual=True):
        super(ST_GCN_block, self).__init__()

        self.gcn = GraphConvolution(in_channels, out_channels, A, cuda_)
        self.tcn = TemporalConvolution(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConvolution(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        """
        x: (batch * people, channels, times, nodes)
        """
        # Graph convolution -> time convolution
        x = self.tcn(self.gcn(x)) + self.residual(x)
        return self.relu(x)


class MarcSTGCN(nn.Module):
    def __init__(self, num_class, num_point, num_person, in_channels, graph, cuda_=torch.cuda.is_available()):
        super(MarcSTGCN, self).__init__()

        self.graph = graph

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        weights_init(self.data_bn, bs=1)

        self.layers = nn.ModuleDict(
            {'layer1': ST_GCN_block(in_channels, 64, A, cuda_, residual=False),
            #  'layer2': ST_GCN_block(64, 64, A, cuda_),
            #  'layer3': ST_GCN_block(64, 64, A, cuda_),
             'layer4': ST_GCN_block(64, 64, A, cuda_),
             'layer5': ST_GCN_block(64, 128, A, cuda_, stride=2),
            #  'layer6': ST_GCN_block(128, 128, A, cuda_),
            #  'layer7': ST_GCN_block(128, 128, A, cuda_),
             'layer8': ST_GCN_block(128, 256, A, cuda_, stride=2),
            #  'layer9': ST_GCN_block(256, 256, A, cuda_),
            #  'layer10': ST_GCN_block(256, 256, A, cuda_)
             }
        )

        self.fc = nn.Linear(256, num_class)
        weights_init(self.fc, bs=num_class)

    def forward(self, x):
        """
        X: Shape (batch, channels, time, nodes, people)
        """
        # n -> batch size, C -> channels 3, T -> num frames, V -> num joints, M -> num persons
      
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T) # (batch, people * nodes * channels, times)
        
        x = self.data_bn(x) # batchnorm
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V) # (batch * people, channels, times, nodes)
        for i in range(len(self.layers)):
            x = self.layers['layer' + str(i+1)](x)
        # N*M,C,T,V

        c_new = x.size(1) # infer new channel size
        x = x.view(N, M, c_new, -1) # (batch, people, new_channel_size, times * nodes)
        x = x.mean(3).mean(1) # Take mean across times*nodes and people
        # return softmax(self.fc(x), dim=1) # in shape: (batch, new_channel_size)
        return self.fc(x)

class STGCN:
    def __init__(self, ds, loss_fn=torch.nn.CrossEntropyLoss()):
        self.train_set, self.test_set, self.val_set = ds
        self.time_steps = self.train_set.X.size()[2]
        ds = self.train_set
        self.n_classes = ds.n_classes
        self.n_point = ds.n_point
        self.num_person = 1
        self.in_channels = 2
        self.loss_fn = loss_fn

        self.graph = MediapipeGraph(self.n_point, ds.in_edge)

        self.classifier = MarcSTGCN(self.n_classes, self.n_point, self.num_person, self.in_channels, self.graph).to(device)
        self.tracking_metrics = {
            "precision": lambda x,y: precision(x, y, 'multiclass', num_classes=self.n_classes),
            "recall": lambda x,y: recall(x, y, 'multiclass', num_classes=self.n_classes),
        }
        self.train()
        # print(train_data.dtype)
        # quit()

        

    def _train_step(self, sample, optimizer, metrics, val_sample=None):
        metrics["scalar"] = {}
        if val_sample is not None:
            val_metrics = {}
        val_metrics = {}
        X, y = sample
        y = y.to(device)
        optimizer.zero_grad()
        outputs = self.classifier(X)
        predictions = torch.nn.functional.one_hot(outputs.argmax(axis=1), num_classes=self.n_classes)
        loss = self.loss_fn(outputs, y)

        loss.backward()

        optimizer.step()
        metrics["scalar"]['loss'] = loss.item()
        for k in self.tracking_metrics.keys():
            metrics["scalar"][k] = self.tracking_metrics[k](y, predictions).item()
        cm = confusion_matrix(y.cpu().numpy().argmax(axis=1), predictions.cpu().numpy().argmax(axis=1))
        if "conf_mat" not in metrics["image"].keys():
          metrics["image"]["conf_mat"] = cm

        else:
            metrics["image"]["conf_mat"][:cm.shape[0], :cm.shape[1]]  += cm
                    
        return metrics

    def _val_step(self, sample, val_metrics):
        val_metrics["scalar"] = {}
        X, y = sample
        y = y.to(device)
        
        with torch.no_grad():
            val_out = self.classifier(X)
            predictions = torch.nn.functional.one_hot(val_out.argmax(axis=1), num_classes=self.n_classes)
            val_loss = self.loss_fn(val_out, y)
            val_metrics["scalar"]["val_loss"] = val_loss.item()
            for k in self.tracking_metrics.keys():
                val_metrics["scalar"]["val_" + k] = self.tracking_metrics[k](y, predictions).item()

        cm = confusion_matrix(y.cpu().numpy().argmax(axis=1), predictions.cpu().numpy().argmax(axis=1))
        if "val_conf_mat" not in val_metrics["image"]:
            val_metrics["image"]["val_conf_mat"]  = cm

        else:
            # print(val_metrics["image"]["val_conf_mat"].shape, cm.shape)
            val_metrics["image"]["val_conf_mat"][:cm.shape[0], :cm.shape[1]]  += cm

        

        return val_metrics


    def train(self, test_split=0.01, val_split=0.3, optimizer=None, lr=0.001, momentum=0.9, epochs=1000, batch_size=32):
        # train_samples = int((1 - test_split) * len(self.ds))
        # test_samples = int(len(self.ds) - train_samples)
        # val_samples = int(val_split * train_samples)
        # train_samples = int(train_samples - val_samples)
        # self.train_set, self.test_set, self.val_set = torch.utils.data.random_split(self.ds, [train_samples, test_samples, val_samples])
        # self.val_set = self.val_set.to(device)

        # quit()
        self.train_set = DataLoader(self.train_set, batch_size=batch_size)
        self.val_set = DataLoader(self.val_set, batch_size=batch_size)



        if optimizer is None:
            # optimizer = torch.optim.SGD(self.classifier.parameters(), lr=lr, momentum=momentum)
            optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr, weight_decay=1e-3)

        writer = SummaryWriter()

        # for epoch in range(epochs):
        #     train_iter = iter(self.train_set)
        #     curr_sample = next(train_iter)
        #     while curr_sample is not None:
        #         self._train_step(curr_sample, optimizer)
        #         curr_sample = next(train_iter)

        print(f"Training with {self.time_steps} time steps")
        for epoch in range(epochs):
            print()
            print(f"Epoch #{epoch + 1}: ")
            train_iter = tqdm(iter(self.train_set))
            scalar_metrics = {}
            train_metrics = {"scalar": {}, "image": {}}
            for idx, curr_sample in enumerate(train_iter):
                train_metrics = self._train_step(curr_sample, optimizer, train_metrics)
                for k in train_metrics["scalar"].keys():
                    if k not in scalar_metrics.keys():
                        scalar_metrics[k] = 0
                    
                    scalar_metrics[k] += train_metrics["scalar"][k] / len(train_iter)
                
                train_iter.set_postfix(train_metrics["scalar"])
            sns.heatmap(train_metrics["image"]["conf_mat"])
            plt.xlabel("Predicted")
            plt.ylabel("True")
            writer.add_figure("Training Confusion Matrix", plt.gcf(), epoch)
            plt.close()
            print()
            print("Validation:")
            val_metrics = {"scalar": {}, "image": {}}
            val_iter = tqdm(iter(self.val_set))
            for idx, val_sample in enumerate(val_iter):
                val_metrics = self._val_step(val_sample, val_metrics)
                for k in val_metrics["scalar"].keys():
                    if k not in scalar_metrics.keys():
                        scalar_metrics[k] = 0
                    scalar_metrics[k] += val_metrics["scalar"][k] / len(val_iter)
                val_iter.set_postfix(val_metrics["scalar"])
                
            # fig = plt.figure()
            # image = torch.image.decode_png(fig.getvalue(), channels=4)

            sns.heatmap(val_metrics["image"]["val_conf_mat"])
            plt.xlabel("Predicted")
            plt.ylabel("True")
            # plt.imshow(hm)
            # quit()
            # hm = fig
            # img_flat = np.frombuffer(fig.canvas.draw().tostring_rgb(), dtype='uint8')
            # image = img_flat.reshape(*reversed(img_flat.get_width_height), 3)
            writer.add_figure("Validation Confusion Matrix", plt.gcf(), epoch)
            plt.close()
            for k in scalar_metrics.keys():
                v = scalar_metrics[k]
                print(f"{k.capitalize()}: {v:.3f}")
                writer.add_scalar(k.capitalize(), v, epoch)


