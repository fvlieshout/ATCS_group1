import torch
from models.pure_graph_encoder import PureGraphEncoder
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv


class RobertaGraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RobertaGraphEncoder, self).__init__()

        self.linlay = nn.Linear(300, input_dim)
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)

    def forward(self, data, mode):
        edge_index, edge_weight = data.edge_index, data.edge_attr
        doc_feats = data.doc_features
        word_feats = data.word_features
        word_feats = self.linlay(word_feats)
        x = torch.cat((doc_feats, word_feats))

        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)

        if mode == 'train':
            mask = data.train_mask
        elif mode == 'val':
            mask = data.val_mask
        elif mode == 'test':
            mask = data.test_mask
        else:
            raise ValueError("Mode '%s' is not supported in forward of graph classifier." % mode)

        return x[mask], data.y[mask]
