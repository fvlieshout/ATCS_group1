import abc
import random
from collections import defaultdict

import torch
import torch.utils.data as data
from data_prep.graph_utils import tf_idf_mtx, get_PMI
from nltk.corpus import reuters
from datasets import load_dataset
from torch_geometric.data import Data


class Dataset(data.Dataset, metaclass=abc.ABCMeta):
    """
    Parent class for all datasets.
    Require to implement a num_classes property.
    """

    @property
    @abc.abstractmethod
    def num_classes(self):
        """
        Return the number of unique classes in the dataset.
        """
        raise NotImplementedError


class TextDataset(Dataset):
    """
    Parent class for text datasets.
    Require to implement get_collate_fn.
    """

    @abc.abstractmethod
    def get_collate_fn(self):
        """
        Return a function (collate_fn) to be used to preprocess a batch in the Dataloader.
        """
        raise NotImplementedError


class GraphDataset(Dataset):
    """
    Parent class for graph datasets.
    Provide methods to generate the edges and the training, validation, and test masks.
    """

    def __init__(self, device, n_train_docs):
        self.device = device
        self.n_train_docs = n_train_docs

        # all initialized later in 'initialize_data'
        self.iton = None
        self.ntoi = None
        self.itol = None
        self.loti = None
        self.data = None
        self.n_words = None

    def initialize_data(self, docs, classes):
        train_docs, test_docs, val_docs = docs
        all_docs = train_docs + test_docs + val_docs

        corpus = self.pre_process_words(all_docs)

        print('Compute tf.idf')
        tf_idf, words = tf_idf_mtx(corpus)
        self.n_words = len(words)

        print('Compute PMI scores')
        pmi_score = get_PMI(corpus)

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

        # Generate masks/splits
        print('Generate masks')
        train_mask, val_mask, test_mask = self.generate_masks(len(train_docs), len(val_docs), len(test_docs))

        # Feature matrix is Identity (according to TextGCN)
        print('Generate feature matrix')
        node_feats = torch.eye(len(self.iton), device=self.device).float()
        # node_feats = torch.rand(size=(len(self.iton), 100), device=self.device).float()
        print('Features mtx is {} GBs in size'.format(node_feats.nelement() * node_feats.element_size() * 1e-9))

        ntol = torch.tensor(self.get_label_node_mapping(), device=self.device)

        # Create pytorch geometric format data
        self.data = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_attr, y=ntol)
        self.data.train_mask = train_mask
        self.data.val_mask = val_mask
        self.data.test_mask = test_mask

    def generate_edges(self, num_docs, tf_idf, pmi_scores):
        """
        Generates edge list and weights based on tf.idf and PMI.

        Args:
            num_docs (int): Number of all documents in the dataset
            tf_idf (SparseMatrix): sklearn Sparse matrix object containing tf.idf values
            pmi_scores (dict): Dictionary of word pairs and corresponding PMI scores

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
        """
        Generates masking for the different splits in the dataset.

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

    @abc.abstractmethod
    def get_label_node_mapping(self):
        """
        Creates the labels to node mapping, where word nodes get the label of -1 and returns it.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def pre_process_words(self, all_docs):
        """
        Preprocesses words respectively to the specific dataset. Must be overwritten by deriving classes.
        """
        raise NotImplementedError


class Reuters(Dataset):
    """
    Parent class for Reuters datasets.
    Provide a method to generate the training, validation, and test documents.
    """

    @staticmethod
    def prepare_reuters(r8=False, val_size=0.1):
        """
        Filters out all documents which have more or less than 1 class.
        Then filters out all classes which have no remaining documents.
        Args:
            r8 (bool, optional): R8 is constructed by taking only the top 10 (original) classes. Defaults to False.
            val_size (float, optional): Proportion of training documents to include in the validation set.
        Returns:
            doc_splits (tuple): Tuple containing 3 List of training, test, and validation documents.
            unique_classes (List): List of Strings containing the class names sorted in alphabetical order.
        """
        # Filter out docs which don't have exactly 1 class
        data = defaultdict(lambda: {'train': [], 'test': []})

        for doc in reuters.fileids():
            categories = reuters.categories(doc)
            if len(categories) == 1:
                if doc.startswith('training'):
                    data[categories[0]]['train'].append(doc)
                if doc.startswith('test'):
                    data[categories[0]]['test'].append(doc)

        # Filter out classes which have no remaining docs
        for cat in reuters.categories():
            if len(data[cat]['train']) < 1 or len(data[cat]['test']) < 1:
                data.pop(cat, None)

        if r8:
            # Choose top 10 classes and then select the ones which still remain after filtering
            popular = sorted(reuters.categories(), key=lambda clz: len(reuters.fileids(clz)), reverse=True)[:10]
            data = dict([(cls, splits) for (cls, splits) in data.items() if cls in popular])

        # Create splits
        train_docs = [doc for cls, splits in data.items() for doc in splits['train']]
        test_docs = [doc for cls, splits in data.items() for doc in splits['test']]

        # Select the validation documents out of the training documents
        val_size = int(len(train_docs) * val_size)
        random.shuffle(train_docs)
        val_docs = train_docs[:val_size]
        train_docs = train_docs[val_size:]

        # sort the unique classes to ensure constant order
        unique_classes = sorted(data.keys())
        return (train_docs, test_docs, val_docs), unique_classes

class HuggingFaceDatasetText(TextDataset):
    """
    Parent class HuggingFace text datasets. To use this class, 
    the underlying HuggingFace dataset must have a text and a label attribute.
    """
    def __init__(self, texts, labels, classes, tokenizer):
        self.texts = texts
        self.labels = labels
        self.classes = classes
        self.tokenizer = tokenizer
    
    @classmethod
    def splits(cls, tokenizer, dataset_name="ag_news", val_size=0.1):
        """
        Creates the train, test, and validation splits for HuggingFace dataset.
        Args:
            dataset_name (String): Name of the hugging face dataset (https://huggingface.co/datasets)
            tokenizer (Tokenizer): Hugging Face tokenizer to encode the 3 dataset splits.
            val_size (float, optional): Proportion of training sample to include in the validation set.
        Returns:
            train_split (TextDataset): Training split.
            test_split (TextDataset): Test split.
            val_split (TextDataset): Validation split.
        """
        dataset = load_dataset(dataset_name)
        train_val_splits = dataset["train"].train_test_split(test_size=val_size)

        train_split = cls.get_split(tokenizer, train_val_splits["train"])
        val_split = cls.get_split(tokenizer, train_val_splits["test"])
        test_split = cls.get_split(tokenizer, dataset["test"])
        
        return train_split, test_split, val_split

    
    @classmethod
    def get_split(cls, tokenizer, dataset):
        texts, labels = cls._prepare_split(dataset)
        unique_cls = dataset.features["label"].names
        return cls(texts, labels, unique_cls, tokenizer)
    
    @staticmethod
    def _prepare_split(dataset):
        texts = []
        labels = []
        for data in dataset:
            texts.append(data["text"])
            labels.append(data["label"])

        return texts, labels
    
    def get_collate_fn(self):
        def collate_fn(batch):
            texts = [data["text"] for data in batch]
            labels = [data["label"] for data in batch]
            encodings = self.tokenizer(texts, truncation=True, padding=True)
            
            items = {key: torch.tensor(val) for key, val in encodings.items()}
            items["labels"] = torch.tensor(labels)

            return items
        return collate_fn
    
    @property
    def num_classes(self):
        return len(self.classes)
    
    def __getitem__(self, idx):
        item = {
            "text": self.texts[idx],
            "label": self.labels[idx]
        }
        return item
    
    def __len__(self):
        return len(self.labels)

