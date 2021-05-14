import abc

from data_prep.dataset import Dataset


class GraphDataset(Dataset):
    """
    Parent class for graph datasets.
    Require to implement a preprocess and generate_features methods.
    """
    def __init__(self, *args, **kwargs):
        # preprocess text
        # compute tf.idf and pmi
        # generate edges
        # generate features
        # build data object
        raise NotImplementedError

    @property
    def num_classes(self):
        """
        Return the number of unique classes in the dataset.
        """
        raise NotImplementedError

    def labels(self):
        """
        Return the labels of datapoints.
        """
        # doc labels and then -1s for all words
        raise NotImplementedError

    def generate_edges(self, tf_idf, tf_idf_words, pmi_scores):
        """
        Generates edge list and weights based on tf.idf and PMI.
        Args:
            tf_idf (SparseMatrix): sklearn Sparse matrix object containing tf.idf values.
            tf_idf_words (list): List of words according to the tf.idf matrix.
            pmi_scores (dict): Dictionary of word pairs and corresponding PMI scores.
        Returns:
            edge_index (Tensor): List of edges.
            edge_attr (Tensor): List of edge weights.
        """
        raise NotImplementedError

    def generate_masks(self, train_num, val_num, test_num):
        """
        Generates masking for the different splits in the dataset.
        Args:
            train_num (int): Number of training documents.
            val_num (int): Number of validation documents.
            test_num (int): Number of test documents.
        Returns:
            train_mask (Tensor): Training mask as boolean tensor.
            val_mask (Tensor): Validation mask as boolean tensor.
            test_mask (Tensor): Test mask as boolean tensor.
        """
        raise NotImplementedError

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

