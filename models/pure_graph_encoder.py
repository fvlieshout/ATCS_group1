import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, GraphConv


class PureGraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, graph_layer_name):
        """
        Creates a GraphEncoder object
        Args:
            input_dim (int): input dimension for the encoder
            hidden_dim (int): hidden dimension that will be used by the encoder
            graph_layer_name (String): the type of GNN layer ('GCNConv' or 'GraphConv')

        Raises:
            Exception: if they graph_layer_name is not in ['GCNConv','GraphConv']
        """
        super(PureGraphEncoder, self).__init__()
        if graph_layer_name == 'GCNConv':
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        elif graph_layer_name == 'GraphConv':
            self.conv1 = GraphConv(input_dim, hidden_dim, aggr='mean')
            self.conv2 = GraphConv(hidden_dim, hidden_dim, aggr='mean')
        else:
            raise Exception('Layer name is not valid: %i' % graph_layer_name)

    def forward(self, data, mode):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
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
