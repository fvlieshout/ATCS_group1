from nltk.tokenize.regexp import WordPunctTokenizer
import torch
from torch_geometric.data import Data

from data_prep.graph_utils import pmi, tf_idf_mtx
from data_prep.dataset import GraphDataset
from datasets import load_dataset


class AGNewsGraph(GraphDataset):
    def __init__(self, device, val_size=0.1, train_doc=None):
        """
        Creates the train, test, and validation splits for AGNews.
        Args:
            device (Device): Device to use to store the dataset.
            val_size (float, optional): Proportion of training documents to include in the validation set.
            train_doc (int, optional): Number of documents to use from the training set. If None, include all.
        """
        self.device = device
        self.train_doc = train_doc

        print('Prepare AGNews dataset')
        docs, labels, classes = self.prepare_agnews(val_size, train_doc)
        train_docs, test_docs, val_docs = docs
        train_labels, test_labels, val_labels = labels

        all_docs = train_docs + test_docs + val_docs
        all_labels = train_labels + test_labels + val_labels

        tokenizer = WordPunctTokenizer() # TODO: look into a better one?
        corpus = [tokenizer.tokenize(sentence) for sentence in all_docs]

        print('Compute tf.idf')
        tf_idf, words = tf_idf_mtx(corpus)

        print('Compute PMI scores')
        pmi_score = pmi(corpus)

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
        ntol = all_labels + [-1] * len(words)
        ntol = torch.tensor(ntol, device=device)

        # Generate masks/splits
        print('Generate masks')
        train_mask, val_mask, test_mask = self.generate_masks(len(train_docs), len(val_docs), len(test_docs))

        # Feature matrix is Identity (according to TextGCN)
        print('Generate feature matrix')
        node_feats = torch.eye(len(self.iton), device=self.device).float()
        print('Features mtx is {} GBs in size'.format(node_feats.nelement() * node_feats.element_size() * 1e-9))

        # Create pytorch geometric format data
        self.data = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_attr, y=ntol)
        self.data.train_mask = train_mask
        self.data.val_mask = val_mask
        self.data.test_mask = test_mask

    @staticmethod
    def prepare_agnews(val_size=0.1, train_doc=None):
        """
        Return the training, validation, and tests splits along with the classes of AGNews dataset.
        Args:
            val_size (float, optional): Proportion of training documents to include in the validation set.
            train_doc (int, optional): Number of documents to use from the training set. If None, include all.
        Returns:
            splits (tuple): Tuple containing 3 list of strings for training, test, and validation datasets.
            labels (tuple): Tuple containing 3 list of labels for training, test, and validation datasets.
            unique_classes (List): List of Strings containing the class names.
        """
        # download the dataset
        dataset = load_dataset("ag_news")

        # create the training and validation splits
        train_val_splits = dataset["train"].train_test_split(test_size=val_size)

        # extract the text for each news article
        train_texts, train_labels = zip(*[(data["text"], data["label"]) for data in train_val_splits["train"]])
        val_texts, val_labels = zip(*[(data["text"], data["label"]) for data in train_val_splits["test"]])
        test_texts, test_labels = zip(*[(data["text"], data["label"]) for data in dataset["test"]])

        unique_classes = train_val_splits["train"].features["label"].names

        if train_doc is not None:
            # For testing with only a few docs:
            test_val_num_docs = int(val_size * train_doc)
            docs = (list(train_texts[:train_doc]), list(test_texts[:test_val_num_docs]), list(val_texts[:test_val_num_docs]))
            labels = (list(train_labels[:train_doc]), list(test_labels[:test_val_num_docs]), list(val_labels[:test_val_num_docs]))
        else:
            docs = (list(train_texts), list(test_texts), list(val_texts))
            labels = (list(train_labels), list(test_labels), list(val_labels))

        return docs, labels, unique_classes

    @property
    def num_classes(self):
        return len(self.itol)
