import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv


class RobertaGraphEncoder(PureGraphEncoder):
    def __init__(self, input_dim, hidden_dim):
        super(RobertaGraphEncoder, self).__init__(input_dim, hidden_dim)
        self.linlay = nn.Linear(300, input_dim)

    def forward(self, data, mode):
        # the batch is a Data object here
        doc_feats = data.doc_features
        word_feats = data.word_features
        word_feats = self.linlay(word_feats)
        x = torch.cat((doc_feats, word_feats))
        return super.forward(x, mode)
