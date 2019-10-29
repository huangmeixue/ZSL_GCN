import torch.nn as nn
import torch.nn.functional as F
from layer import GraphConvolution


class GCN(nn.Module):
    def __init__(self, in_feature, hidden_size, out_feature, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(in_feature, hidden_size)
        self.activate = nn.LeakyReLU(0.2, inplace=True)
        self.gc2 = GraphConvolution(hidden_size, out_feature)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = F.dropout(self.activate(x), self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x