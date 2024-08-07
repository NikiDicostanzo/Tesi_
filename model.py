from dgl.nn import GATv2Conv, SAGEConv
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import dgl
import dgl.function as fn

    
#### NODE 
# Contruct a two-layer GNN model
class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')      #, norm=nn.BatchNorm1d(hid_feats))#'mean') #'lstm')#pool
        self.conv2 = SAGEConv(
            in_feats=hid_feats, out_feats=hid_feats, aggregator_type='mean')        #, norm=nn.BatchNorm1d(out_feats))#'mean')
        self.conv3 = SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')
        
    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        h = F.relu(h)
        h = self.conv3(graph, h)
        return h

#### +++ EDGE

class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(torch.cat([h_u, h_v], 1))
        return {'score': score}

    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            scores = graph.edata['score']
            return scores#, F.log_softmax(scores, dim=1)

class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.sage = SAGE(in_features, hidden_features, out_features)
        self.pred = MLPPredictor(out_features, out_features)

    def forward(self, g, x):
        h = self.sage(g, x)
        return self.pred(g, h)
