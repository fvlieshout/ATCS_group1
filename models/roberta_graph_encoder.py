import torch
from models.pure_graph_encoder import PureGraphEncoder
from torch import nn


class RobertaGraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, graph_layer_name):
        super(RobertaGraphEncoder, self).__init__()

        self.linlay = nn.Linear(300, input_dim)
        self.gnn_encoder = PureGraphEncoder(input_dim, hidden_dim, graph_layer_name)

    def forward(self, data, mode):
        doc_feats = data.doc_features
        word_feats = data.word_features
        word_feats = self.linlay(word_feats)
        data.x = torch.cat((doc_feats, word_feats))
        return self.gnn_encoder(data, mode)
