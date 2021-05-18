import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GraphConv


class PureGraphEncoder(nn.Module):
    def __init__(self, doc_dim=10000, word_dim=300, output_dim=768):
        """
        Initializes the pure graph encoder model

        Args:
            doc_dim (int, optional): Dimension of document embeddings. Defaults to 10000.
            word_dim (int, optional): Dimension of word embeddings. Defaults to 300.
        """
        super(PureGraphEncoder, self).__init__()
        self.linlay = nn.Linear(doc_dim, word_dim)
        self.conv1 = GraphConv(word_dim, output_dim)
        self.conv2 = GraphConv(output_dim, output_dim)

    def forward(self, data, mode):
        edge_index, edge_weight = data.edge_index, data.edge_attr
        word_feats, doc_feats = data.word_features, data.doc_features

        doc_feats = self.linlay(doc_feats)
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
