import torch
from models.pure_graph_encoder import PureGraphEncoder
from torch import nn


class GloveGraphEncoder(nn.Module):
    def __init__(self, doc_dim=10000, word_dim=300, hidden_dim=768, graph_layer_name='GraphConv'):
        super().__init__()

        self.linlay = nn.Linear(doc_dim, word_dim)
        self.gnn_encoder = PureGraphEncoder(word_dim, hidden_dim, graph_layer_name)

    def forward(self, data, mode):
        doc_feats = data.doc_features
        word_feats = data.word_features
        doc_feats = self.linlay(doc_feats)
        data.x = torch.cat((doc_feats, word_feats))
        return self.gnn_encoder(data, mode)
