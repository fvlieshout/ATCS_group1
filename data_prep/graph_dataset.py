import abc

import torch
from torch_geometric.data import Dataset as GeometricDataset
from torch_geometric.data import Data
from transformers import RobertaTokenizerFast

from data_prep.dataset import Dataset
from data_prep.graph_utils import tf_idf_mtx, get_PMI


class GraphDataset(Dataset, GeometricDataset):
    """
    Parent class for graph datasets.
    Require to implement a preprocess and generate_features methods.
    """
    def __init__(self, corpus, device, *args, **kwargs):
        self.device = device
        self.roberta_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.num_classes = corpus.num_classes

        train_texts, train_labels = corpus.train_data()
        val_texts, val_labels = corpus.val_data()
        test_texts, test_labels = corpus.test_data()

        self.raw_texts = train_texts + val_texts + test_texts
        # Hopefully no tokenizer makes a token "doc.i"
        all_docs = ['doc.{}'.format(i) for i in range(len(self.raw_text))]

        print('Preprocess corpus')
        tokenized_text, self.tokens = self.preprocess(self.raw_text)

        iton = list(all_docs + self.tokens)
        ntoi = {iton[i]: i for i in range(len(iton))}

        print('Compute tf.idf')
        tf_idf, tf_idf_words = tf_idf_mtx(tokenized_text)

        print('Compute PMI scores')
        pmi_score = get_PMI(tokenized_text)

        print('Generate edges')
        edge_index, edge_attr = self.generate_edges(tf_idf, tf_idf_words, pmi_score, ntoi)

        print('Generate masks')
        train_mask, val_mask, test_mask = self.generate_masks(len(corpus.train_doc_ids), len(corpus.val_doc_ids),
                                                              len(corpus.test_doc_ids), len(iton))

        labels = self.labels(train_labels + val_labels + test_labels, len(iton))

        self.data = Data(edge_index=edge_index, edge_attr=edge_attr, y=labels)
        self.data.train_mask = train_mask
        self.data.val_mask = val_mask
        self.data.test_mask = test_mask

    # THIS FUNCTION IS NOT REALLY NEEDED
    @property
    def num_classes(self):
        """
        Return the number of unique classes in the dataset.
        """
        return self.num_classes

    def labels(self, doc_labels, all_num):
        """
        Return the labels of datapoints.

        Args:
            doc_labels (List): List of document labels.
            all_num (int): Number of all nodes, including words
        Returns:
            labels (Tensor): Document and word labels
        """
        labels = torch.zeros(all_num) - 1
        labels[:len(doc_labels)] = torch.tensor(doc_labels)
        return labels

    def generate_edges(self, tf_idf, tf_idf_words, pmi_scores, ntoi):
        """
        Generates edge list and weights based on tf.idf and PMI.
        Args:
            tf_idf (SparseMatrix): sklearn Sparse matrix object containing tf.idf values.
            tf_idf_words (list): List of words according to the tf.idf matrix.
            pmi_scores (dict): Dictionary of word pairs and corresponding PMI scores.
            ntoi (dict): Dictionary mapping from nodes to indices.
        Returns:
            edge_index (Tensor): List of edges.
            edge_attr (Tensor): List of edge weights.
        """
        edge_index = []
        edge_attr = []

        # Document-word edges
        for d_ind, doc in enumerate(tf_idf):
            tf_idf_inds = doc.indices
            for tf_idf_ind in tf_idf_inds:
                # Convert index from tf.idf to index in ntoi
                word = tf_idf_words[tf_idf_ind]
                w_ind = ntoi[word]

                edge_index.append([d_ind, w_ind])
                edge_index.append([w_ind, d_ind])
                edge_attr.append(tf_idf[d_ind, tf_idf_ind])
                edge_attr.append(tf_idf[d_ind, tf_idf_ind])

        # Word-word edges
        for (word_i, word_j), score in pmi_scores.items():
            w_i_ind = ntoi[word_i]
            w_j_ind = ntoi[word_j]
            edge_index.append([w_i_ind, w_j_ind])
            edge_index.append([w_j_ind, w_i_ind])
            edge_attr.append(score)
            edge_attr.append(score)

        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_attr = torch.tensor(edge_attr).float()
        return edge_index, edge_attr

    def generate_masks(self, train_num, val_num, test_num, all_num):
        """
        Generates masking for the different splits in the dataset.
        Args:
            train_num (int): Number of training documents.
            val_num (int): Number of validation documents.
            test_num (int): Number of test documents.
            all_num (int): Number of all nodes, including words
        Returns:
            train_mask (Tensor): Training mask as boolean tensor.
            val_mask (Tensor): Validation mask as boolean tensor.
            test_mask (Tensor): Test mask as boolean tensor.
        """
        train_mask = torch.zeros(all_num, device=self.device)
        train_mask[:train_num] = 1

        val_mask = torch.zeros(all_num, device=self.device)
        val_mask[train_num:train_num + val_num] = 1

        # Mask all non-test docs
        test_mask = torch.zeros(all_num, device=self.device)
        test_mask[val_num + train_num:val_num + train_num + test_num] = 1

        return train_mask.bool(), val_mask.bool(), test_mask.bool()

    @abc.abstractmethod
    def preprocess(self):
        """
        Preprocesses the corpus.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def generate_features(self):
        """
        Generates node features.
        """
        raise NotImplementedError

    def len(self):
        return 1

    def get(self, idx):
        return self.data
