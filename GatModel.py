from dgl.nn import GATv2Conv
import torch.nn.functional as F
import torch.nn as nn
from predictor import DotProductPredictor

""" class GATModel(nn.Module):
   def __init__(self, input_dim, hidden_dim, output_dim, num_heads):
       super(GATModel, self).__init__()
       self.conv1 = GATv2Conv(input_dim, hidden_dim, num_heads)
       self.conv2 = GATv2Conv(hidden_dim * num_heads, output_dim, num_heads)

   def forward(self, g, h):
       h = self.conv1(g, h)
       h = F.relu(h)
       h = self.conv2(g, h)
       return h """

class GATModel(nn.Module):
    def __init__(self, in_features, out_features, num_heads):
        super().__init__()
        self.gat = GATv2Conv(in_features, out_features, num_heads)
        self.pred = DotProductPredictor()

    def forward(self, g, x):
        h = self.gat(g, x)
        return self.pred(g, h)
    
# class Model(nn.Module):
#     def __init__(self, in_features, hidden_features, out_features):
#         super().__init__()
#         self.sage = SAGE(in_features, hidden_features, out_features)
#         self.pred = DotProductPredictor()
#     def forward(self, g, x):
#         h = self.sage(g, x)
#         return self.pred(g, h)