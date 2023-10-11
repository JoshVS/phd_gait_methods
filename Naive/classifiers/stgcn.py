import torch
import torch_geometric as G
import torch.nn as nn


class TemporalGatedConv(nn.Module):

    def __init__(self, time_channels):

        
        super(TemporalGatedConv, self).__init__()
        self.conv = nn.Conv1d(time_channels, 64)
    
    def forward(self, X):
        """
        X is shape (n, time, nodes, features)
        """
        inp = X.permute(0, 2, 3, 1)
        return nn.functional.glu(self.conv(inp)).permute(0, 3, 1, 2)
    





class STGCN(nn.Module):

    def __init__(self, time_channels, spatial_channels, out_channels,
                 num_nodes):
        
        super(STGCN, self).__init__()
        self.time1 = TemporalGatedConv(time_channels)
        self.convblock = G.torch_geometric.nn.conv.GraphConv(64, 16)        
        self.time2 = TemporalGatedConv(16)

    def forward(self, X):
        x1 = self.time1(X)
        x1 = torch.flatten(x1, start_dim=0, end_dim=1)
        x2 = self.convblock(x1)
        x2 = torch.unflatten(x1, dim=0, sizes=(-1, 16))
        x3 = self.time2(x2)
        return x3
