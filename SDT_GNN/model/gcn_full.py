import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from SDT_GNN.utils import utils


class GCN_Full(nn.Module):
    """
    An implementation of full batch GCN.
    For details about the algorithm see this paper:
    "Semi-Supervised Classification with Graph Convolutional Networks"

    Args:
        in_feats (int): Dimension of input features.
        n_hidden (int): Dimension of hidden layers.
        n_layer (int): Number of hidden layers.
        n_classes (int): number of classes.
        dropout (float): Dropout rate.
        activation (str): 'relu', 'sigmoid', 'tanh', 'leaky_relu'.
    """
    
    def __init__(self, 
                 in_feats: int = 32, 
                 n_hidden: int = 32, 
                 n_layers: int = 2, 
                 n_classes: int = 32, 
                 dropout: float = 0.0, 
                 activation: str = 'relu'):
        super().__init__()
        
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_classes = n_classes
        
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        elif activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation == None:
            self.activation = activation
        else:
            raise NotImplementedError('No Support for \'{}\' Yet. Please Try Different Activation Functions.'.format(activation))
        
        self.layers = nn.ModuleList()
        self.layers.append(dgl.nn.GraphConv(self.in_feats, 
                                            self.n_hidden, 
                                            activation=self.activation))
        
        for i in range(self.n_layers-2):
            self.layers.append(dgl.nn.GraphConv(self.n_hidden, 
                                                self.n_hidden, 
                                                activation=self.activation))
        
        self.layers.append(dgl.nn.GraphConv(self.n_hidden, 
                                            self.n_classes))
        
        self.dropout = nn.Dropout(dropout)
    

    def forward(self, graph, x):
        h = self.dropout(x)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h