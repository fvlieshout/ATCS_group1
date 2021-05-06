import torch
from collections import defaultdict
import random

import nltk
import torch_geometric
# nltk.download('reuters')
# from nltk.corpus import reuters
from datasets.reuters_graph import R8, R52
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from datasets.graph_utils import PMI, tf_idf_mtx

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(len(r8.iton), 200)
        self.conv2 = GCNConv(200, 8)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

def eval(model, data, mask):
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = int(pred[mask].eq(data.y[mask]).sum().item())
    acc = correct / int(mask.sum())
    print('Accuracy: {:.4f}'.format(acc))

def train(model, data, mask):
    print()

if __name__ == "__main__":
    r8 = R8(device)
    # cora_dataset = torch_geometric.datasets.Planetoid(root='/tmp/cora', name="Cora")
    model = Net(r8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    data = r8.data

    r8.data.to(device)
    eval(model, data, data.val_mask)
