import dgl
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from SDT_GNN.utils import utils


class ClusterGCN(nn.Module):
    """
    An implementation of ClusterGCN.
    For details about the algorithm see this paper:
    "Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks"

    Args:
        in_feats (int): Dimension of input features.
        n_hidden (int): Dimension of hidden layers.
        n_layer (int): Number of hidden layers.
        n_classes (int): number of classes.
        dropout (float): Dropout rate.
        activation (str): 'relu', 'sigmoid', 'tanh', 'leaky_relu'.
    """
    
    def __init__(self, 
                 in_feats, 
                 n_hidden, 
                 n_layers, 
                 n_classes, 
                 dropout, 
                 activation):
        super().__init__()
        
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.activation = utils.activation_function(activation)
        
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
    

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != 0:
                h = self.dropout(h)
        return h


    def inference(self, 
                  g, 
                  device, 
                  batch_size, 
                  num_workers, 
                  buffer_device=None):
            
        feat = g.ndata['feat']
        
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        
        dataloader = dgl.dataloading.DataLoader(g, 
                                                torch.arange(g.num_nodes()).to(g.device), 
                                                sampler, 
                                                device=device,
                                                batch_size=batch_size, 
                                                shuffle=False, 
                                                drop_last=False, 
                                                num_workers=num_workers)

        if buffer_device is None:
            buffer_device = device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes,
                device=buffer_device, pin_memory=True)
            feat = feat.to(device)
            
            for _, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                y[output_nodes[0]:output_nodes[-1]+1] = h.to(buffer_device)
            feat = y
        
        return y
    
    