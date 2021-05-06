import torch
from collections import defaultdict
import random

import nltk
import torch_geometric
nltk.download('reuters')
from nltk.corpus import reuters
from datasets.reuters_graph import R8, R52
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from datasets.graph_utils import PMI, tf_idf_mtx

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# def prepare_reuters(r8=False):
#     """Filters out all documents which have more or less than 1 class. Then filters out all classes which have no remaining documents.

#     Args:
#         r8 (bool, optional): R8 is constructed by taking only the top 10 (original) classes. Defaults to False.

#     Returns:
#         train_docs (list): List of training documents.
#         test_docs (list): List of test documents.
#     """    
#     # Filter out docs which don't have exactly 1 class
#     data = defaultdict(lambda: {'train': [], 'test': []})
#     for doc in reuters.fileids():
#         # print("reuter field=", doc)
#         if len(reuters.categories(doc)) == 1:
#             if doc.startswith('training'):
#                 data[reuters.categories(doc)[0]]['train'].append(doc)
#             elif doc.startswith('test'):
#                 data[reuters.categories(doc)[0]]['test'].append(doc)
#             else:
#                 print(doc)

#     # Filter out classes which have no remaining docs
#     for cls in reuters.categories():
#         if len(data[cls]['train']) < 1 or len(data[cls]['test']) < 1:
#             data.pop(cls, None)

#     if r8:
#         # Choose top 10 classes and then select the ones which still remain after filtering
#         popular = sorted(reuters.categories(), key=lambda cls: len(reuters.fileids(cls)), reverse=True)[:10]
#         data = dict([(cls, splits) for (cls, splits) in data.items() if cls in popular])

#     # Create splits
#     train_docs = [doc for cls, splits in data.items() for doc in splits['train']]
#     test_docs = [doc for cls, splits in data.items() for doc in splits['test']]

#     return train_docs, test_docs, list(data.keys())

# class Reuters:
#     def __init__(self, r8, device, val_size=0.1):
#         self.device = device
#         print('Prepare Reuters dataset')
#         train_docs, test_docs, classes = prepare_reuters(r8)
#         corpus = [[word.lower() for word in reuters.words(doc)] for doc in train_docs + test_docs]
        
#         print('Compute tf.idf')
#         tf_idf, words = tf_idf_mtx(corpus)
        
#         print('Compute PMI scores')
#         pmi = PMI(corpus)
        
#         # Index to node name mapping
#         self.iton = list(train_docs + test_docs + words)
#         # Node name to index mapping
#         self.ntoi = {self.iton[i]: i for i in range(len(self.iton))}
        
#         # Edge index and values for dataset
#         print('Generate edges')
#         edge_index, edge_attr = self.generate_edges(len(train_docs + test_docs), tf_idf, pmi)
        
#         # Index to label mapping
#         self.itol = classes
#         # Label in index mapping
#         self.loti = {self.itol[i]: i for i in range(len(self.itol))}
#         # Labels to node mapping, where word nodes get the label of -1
#         ntol = [self.loti[reuters.categories(node)[0]] if reuters.categories(node) else -1 for node in self.iton]
#         ntol = torch.tensor(ntol, device=device)
        
#         # Generate masks/splits
#         print('Generate masks')
#         train_mask, val_mask, test_mask = self.generate_masks(len(train_docs), len(test_docs), val_size)
        
#         # Feature matrix is Identity (according to TextGCN)
#         print('Generate feature matrix')
#         node_feats = torch.eye(len(self.iton), device=self.device).float()
#         print('Features mtx is {} GBs in size'.format(node_feats.nelement() * node_feats.element_size() * 1e-9))
        
#         # Create pytorch geometric format data
#         self.data = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_attr, y=ntol)
#         self.data.train_mask = train_mask
#         self.data.val_mask = val_mask
#         self.data.test_mask = test_mask
        
#     def generate_edges(self, num_docs, tf_idf, pmi):
#         edge_index = []
#         edge_attr = []
        
#         # Document-word edges
#         for d_ind, doc in enumerate(tf_idf):
#             word_inds = doc.indices
#             for w_ind in word_inds:
#                 edge_index.append([d_ind, num_docs + w_ind])
#                 edge_index.append([num_docs + w_ind, d_ind])
#                 edge_attr.append(tf_idf[d_ind, w_ind])
#                 edge_attr.append(tf_idf[d_ind, w_ind])
        
#         # Word-word edges
#         for (word_i, word_j), score in pmi.items():
#             w_i_ind = self.ntoi[word_i]
#             w_j_ind = self.ntoi[word_j]
#             edge_index.append([w_i_ind, w_j_ind])
#             edge_index.append([w_j_ind, w_i_ind])
#             edge_attr.append(score)
#             edge_attr.append(score)
        
#         edge_index = torch.tensor(edge_index).t().contiguous()
#         edge_attr = torch.tensor(edge_attr).float()
#         return edge_index, edge_attr
    
#     def generate_masks(self, train_num, test_num, val_size):
#         # Mask all non-training docs
#         train_mask = torch.zeros(len(self.iton), device=self.device)
#         train_mask[:train_num] = 1
        
#         # Randomly select val docs from train-docs and mask accordingly
#         val_mask = torch.zeros(len(self.iton), device=self.device)
#         val_mask_inds = torch.randperm(train_num)[:int(val_size * train_num)]
#         val_mask[val_mask_inds] = 1
#         train_mask[val_mask_inds] = 0
        
#         # Mask all non-test docs
#         test_mask = torch.zeros(len(self.iton), device=self.device)
#         test_mask[train_num:test_num] = 1
#         test_mask = train_mask.bool()

#         return train_mask.bool(), val_mask.bool(), test_mask.bool()
        
        
# class R52(Reuters):
#     def __init__(self, device, val_size=0.1):
#         super().__init__(r8=False, device=device, val_size=val_size)
        
        
# class R8(Reuters):
#     def __init__(self, device, val_size=0.1):
#         super().__init__(r8=True, device=device, val_size=val_size)

class Net(torch.nn.Module):
    def __init__(self, r8):
        super(Net, self).__init__()
        self.conv1 = GCNConv(len(r8.iton), 8)
        self.conv2 = GCNConv(8, 8)

    def forward(self, data):
        x, edge_index, edge_weight = data.x.double(), data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

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
