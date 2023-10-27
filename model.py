import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, init
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PHLP(torch.nn.Module):
    def __init__(self, hidden_channels_PI=1024, num_layers=3, dropout=0.5):
        super(PHLP, self).__init__()
            
        self.linears = torch.nn.ModuleList()
        for i in range(num_layers-1):
            self.linears.append(Linear(hidden_channels_PI, hidden_channels_PI,dtype=torch.float64))
        self.linears.append(Linear(hidden_channels_PI, 1,dtype=torch.float64))
        
        self.dropout = dropout
        
        self.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.linears[:-1]):
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.linears[-1](x)
        
        return torch.sigmoid(x).view(-1)
    
    def reset_parameters(self):
        for lin in self.linears:
            lin.reset_parameters()
            

class Multi_PHLP(torch.nn.Module):
    def __init__(self, num_multi=5, hidden_channels_PI=1024, num_layers=3, dropout=0.5):
        super(Multi_PHLP, self).__init__()
            
        self.PHLPs = torch.nn.ModuleList()
        for i in range(num_multi):
            self.PHLPs.append(PHLP(hidden_channels_PI=hidden_channels_PI, num_layers=num_layers,dropout=dropout))

        self.alpha = torch.nn.Parameter(torch.zeros(num_multi))
        
        self.dropout = dropout
        
        self.reset_parameters()

    def forward(self, x, each_result=False):
        result = torch.zeros((len(x), len(self.PHLPs))).to(device)
        for i, each_PHLP in enumerate(self.PHLPs):
            result[:,i] = each_PHLP(x[:, i])

        alpha = torch.softmax(self.alpha, dim=0)
        out =result*alpha
        out = torch.clamp(out.sum(dim=1), min=0.0, max=1.0)
        
        if each_result:
            return out, result
        else:
            return out
    
    def reset_parameters(self):
        for each_PHLP in self.PHLPs:
            each_PHLP.reset_parameters()
        torch.nn.init.constant_(self.alpha, 0)