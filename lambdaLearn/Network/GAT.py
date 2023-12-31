import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(self, dim_in, num_classes, dim_hidden=16, heads=8, dropout=0.6):
        """
        :param dim_in: Node feature dimension.
        :param dim_hidden: the dimension of hidden layers.
        :param num_classes: Number of classes.
        :param dropout: The dropout rate.
        :param heads: The number of heads.
        """
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(dim_in, dim_hidden, heads, dropout=dropout)
        self.conv2 = GATConv(dim_hidden * heads, num_classes, heads=1, concat=False, dropout=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
