from nltk.tokenize.regexp import WordPunctTokenizer
import torch
from torch_geometric.data import Data

from data_prep.graph_utils import pmi, tf_idf_mtx
from data_prep.dataset import Dataset
from datasets import load_dataset


class AGNewsGraph(Dataset):
    def __init__(self, device, val_size=0.1):
        """
        Creates the train, test, and validation splits for AGNews.
        Args:
            device (Device): Device to use to store the dataset.
            val_size (float, optional): Proportion of training documents to include in the validation set.
        Returns:
            train_split (Dataset): Training split.
            test_split (Dataset): Test split.
            val_split (Dataset): Validation split.
        """
        self.device = device

        print('Prepare AGNews dataset')
        docs, labels, classes = self.prepare_agnews(val_size)
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
        edge_index, edge_attr = self.generate_edges(len(all_docs), tf_idf, pmi_score)

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

    def generate_edges(self, num_docs, tf_idf, pmi_scores):
        """Generates edge list and weights based on tf.idf and PMI.

        Args:
            num_docs (int): Number of all documents in the dataset
            tf_idf (SparseMatrix): sklearn Sparse matrix object containing tf.idf values
            pmi_scores (dict): Dictonary of word pairs and corresponding PMI scores

        Returns:
            edge_index (Tensor): List of edges.
            edge_attr (Tensor): List of edge weights.
        """
        edge_index = []
        edge_attr = []

        # Document-word edges
        for d_ind, doc in enumerate(tf_idf):
            word_inds = doc.indices
            for w_ind in word_inds:
                edge_index.append([d_ind, num_docs + w_ind])
                edge_index.append([num_docs + w_ind, d_ind])
                edge_attr.append(tf_idf[d_ind, w_ind])
                edge_attr.append(tf_idf[d_ind, w_ind])

        # Word-word edges
        for (word_i, word_j), score in pmi_scores.items():
            w_i_ind = self.ntoi[word_i]
            w_j_ind = self.ntoi[word_j]
            edge_index.append([w_i_ind, w_j_ind])
            edge_index.append([w_j_ind, w_i_ind])
            edge_attr.append(score)
            edge_attr.append(score)

        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_attr = torch.tensor(edge_attr).float()
        return edge_index, edge_attr

    def generate_masks(self, train_num, val_num, test_num):
        """Generates masking for the different splits in the dataset.

        Args:
            train_num (int): Number of training documents.
            val_num (int): Number of validation documents.
            test_num (int): Number of test documents.

        Returns:
            train_mask (List): Training mask as boolean list.
            val_mask (List): Validation mask as boolean list.
            test_mask (List): Test mask as boolean list.
        """
        val_mask = torch.zeros(len(self.iton), device=self.device)
        val_mask[:val_num] = 1

        train_mask = torch.zeros(len(self.iton), device=self.device)
        train_mask[val_num:val_num + train_num] = 1

        # Mask all non-test docs
        test_mask = torch.zeros(len(self.iton), device=self.device)
        test_mask[val_num + train_num:val_num + train_num + test_num] = 1

        return train_mask.bool(), val_mask.bool(), test_mask.bool()

    @staticmethod
    def prepare_agnews(val_size=0.1):
        """
        Return the training, validation, and tests splits along with the classes of AGNews dataset.
        Args:
            val_size (float, optional): Proportion of training documents to include in the validation set.
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

         # For testing with only a few docs:
        docs = (list(train_texts[:1000]), list(test_texts[:100]), list(val_texts[:100]))
        labels = (list(train_labels[:1000]), list(test_labels[:100]), list(val_labels[:100]))

        
        return docs, labels, unique_classes
    
    @property
    def num_classes(self):
        return len(self.itol)
