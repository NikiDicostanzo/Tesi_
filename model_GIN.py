from dgl.nn import GATv2Conv, SAGEConv, GINConv
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import dgl
import dgl.function as fn

# Contruct a two-layer GNN model
class GIN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.apply_func = nn.Sequential(
            nn.Linear(in_feats, hid_feats),
            nn.ReLU(),
            nn.Linear(hid_feats, hid_feats)
        )
        self.conv1 = GINConv(self.apply_func, aggregator_type='sum')
        self.conv2 = GINConv(self.apply_func, aggregator_type='sum')
        self.conv3 = GINConv(self.apply_func, aggregator_type='sum')
        
    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        h = F.relu(h)
        h = self.conv3(graph, h)
        return h

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

""" 
class EdgeClassifier(nn.Module):

    def __init__(self, edge_classes, m_layers, dropout, in_chunks, out_chunks, hidden_dim, device, doProject=True):
        super().__init__()

        # Project inputs into higher space
        self.projector = InputProjector(in_chunks, out_chunks, device, doProject)

        # Perform message passing
        m_hidden = self.projector.get_out_lenght()
        self.message_passing = nn.ModuleList()
        self.m_layers = m_layers
        for l in range(m_layers):
            self.message_passing.append(GcnSAGELayer(m_hidden, m_hidden, F.relu, 0.))

        # Define edge predictori layer
        self.edge_pred = MLPPredictor(m_hidden, hidden_dim, edge_classes, dropout)  

    def forward(self, g, h):
        h = self.projector(h)
        for l in range(self.m_layers):
            h = self.message_passing[l](g, h)
        
        e = self.edge_pred(g, h)
        return e """