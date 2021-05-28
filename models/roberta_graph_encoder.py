import torch
from models.pure_graph_encoder import PureGraphEncoder
from torch import nn


class RobertaGraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, graph_layer_name):
        """
        Creates a RobertaGraphEncoder object
        Args:
            input_dim (int): input dimension for the encoder
            hidden_dim (int): hidden dimension that will be used by the encoder
            graph_layer_name (String): the type of GNN layer ('GCNConv' or 'GraphConv')
        """
        super(RobertaGraphEncoder, self).__init__()

        self.linlay = nn.Linear(300, input_dim)
        self.gnn_encoder = PureGraphEncoder(input_dim, hidden_dim, graph_layer_name)

    def forward(self, data, mode):
        doc_feats = data.doc_features
        word_feats = data.word_features
        word_feats = self.linlay(word_feats)
        data.x = torch.cat((doc_feats, word_feats))
        return self.gnn_encoder(data, mode)
