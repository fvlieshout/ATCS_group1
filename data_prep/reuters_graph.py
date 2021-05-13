import nltk

nltk.download('reuters')
from nltk.corpus import reuters
import torch
from torch_geometric.data import Data

from data_prep.graph_utils import tf_idf_mtx, get_PMI
from data_prep.dataset import GraphDataset, Reuters



class ReutersGraph(GraphDataset, Reuters):
    def __init__(self, device, r8=False, val_size=0.1, train_doc=None):
        """
        Creates the train, test, and validation splits for R52 or R8.
        Args:
            device (Device): Device to use to store the dataset.
            r8 (bool, optional): If true, it initializes R8 instead of R52. Defaults to False.
            val_size (float, optional): Proportion of training documents to include in the validation set.
            train_doc (int, optional): Number of documents to use from the training set. If None, include all.
        Returns:
            train_split (Dataset): Training split.
            test_split (Dataset): Test split.
            val_split (Dataset): Validation split.
        """
        self.device = device
        self.train_doc = train_doc

        print('Prepare Reuters dataset')
        (train_docs, test_docs, val_docs), classes = self.prepare_reuters(r8, val_size, train_doc)
        all_docs = train_docs + test_docs + val_docs
        corpus = [[word.lower() for word in reuters.words(doc)] for doc in all_docs]

        print('Compute tf.idf')
        tf_idf, words = tf_idf_mtx(corpus)

        print('Compute PMI scores')
        pmi_score = get_PMI(corpus)

        # Index to node name mapping
        self.iton = list(all_docs + words)
        # Node name to index mapping
        self.ntoi = {self.iton[i]: i for i in range(len(self.iton))}

        # Edge index and values for dataset
        print('Generate edges')
        edge_index, edge_attr = self.generate_edges(tf_idf, words, pmi_score)

        # Index to label mapping
        self.itol = classes
        # Label in index mapping
        self.loti = {self.itol[i]: i for i in range(len(self.itol))}
        # Labels to node mapping, where word nodes get the label of -1
        ntol = [self.loti[reuters.categories(node)[0]] if reuters.categories(node) else -1 for node in self.iton]
        ntol = torch.tensor(ntol, device=device)

        # Generate masks/splits
        print('Generate masks')
        train_mask, val_mask, test_mask = self.generate_masks(len(train_docs), len(val_docs), len(test_docs))

        # Feature matrix is Identity (according to TextGCN)
        print('Generate feature matrix')
        node_feats = torch.eye(len(self.iton), device=self.device).float()
        # node_feats = torch.rand(size=(len(self.iton), 100), device=self.device).float()
        print('Features mtx is {} GBs in size'.format(node_feats.nelement() * node_feats.element_size() * 1e-9))

        # Create pytorch geometric format data
        self.data = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_attr, y=ntol)
        self.data.train_mask = train_mask
        self.data.val_mask = val_mask
        self.data.test_mask = test_mask

    @staticmethod
    def prepare_reuters(r8=False, val_size=0.1, train_doc=None):
        (train_docs, test_docs, val_docs), unique_classes = Reuters.prepare_reuters(r8=r8, val_size=val_size)

        if train_doc is not None:
            # For testing with only a few docs:
            test_val_num_docs = int(val_size * train_doc)
            return (train_docs[:train_doc], test_docs[:test_val_num_docs], val_docs[:test_val_num_docs]), unique_classes
        else:
            return train_docs, test_docs, val_docs, unique_classes

    @property
    def num_classes(self):
        return len(self.itol)


class R52Graph(ReutersGraph):
    """
    Wrapper for the R52 dataset.
    """

    def __init__(self, device, val_size=0.1, train_doc=None):
        super().__init__(r8=False, device=device, val_size=val_size, train_doc=train_doc)


class R8Graph(ReutersGraph):
    """
    Wrapper for the R8 dataset.
    """

    def __init__(self, device, val_size=0.1, train_doc=None):
        super().__init__(r8=True, device=device, val_size=val_size, train_doc=train_doc)
