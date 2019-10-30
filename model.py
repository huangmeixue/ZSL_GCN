import torch.nn as nn
import torch.nn.functional as F
from layer import GraphConvolution


class GCN(nn.Module):
    def __init__(self, input_dim, hiddens_dim, output_dim, dropout):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.feature_size = [input_dim] + hiddens_dim + [output_dim]

        self.layers = nn.ModuleList()
        for i in range(len(self.feature_size)-1):
            self.layers.append(GraphConvolution(input_dim=self.feature_size[i],
                                                output_dim=self.feature_size[i+1]))
        
    def forward(self, input, adj):
        input = F.normalize(input)
        # Build sequential layer model
        self.hiddens = []
        self.hiddens.append(input)
        for i, gcn_layer in enumerate(self.layers):
            activate = lambda x: F.leaky_relu(x, 0.2)
            dropout = lambda x: F.dropout(x, self.dropout[i], training=self.training)
            if i != len(self.layers) - 1:
                hidden = activate(dropout(gcn_layer(self.hiddens[-1], adj)))
            else:
                hidden = dropout(gcn_layer(self.hiddens[-1], adj))
            self.hiddens.append(hidden)
        output = self.hiddens[-1]
        return output

class GCN_simple(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.activate = nn.LeakyReLU(0.2, inplace=True)
        self.gc2 = GraphConvolution(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, input, adj):
        x = self.gc1(x, adj)
        x = F.dropout(self.activate(x), self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x