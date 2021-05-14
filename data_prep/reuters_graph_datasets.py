from collections import defaultdict
import random
import os

import nltk
nltk.download('reuters')
from nltk.corpus import reuters
import torchtext.vocab as vocab
import torch
from tqdm import tqdm
from torch_geometric.data import Data, Dataset
from transformers import RobertaModel
from data_prep.graph_utils import PMI, tf_idf_mtx


class Reuters(Dataset):
    def __init__(self, root, device, r8=False, val_size=0.1, mode='train', tokenizer=None, transform=None, pre_transform=None):
        """
        Creates the train, test, and validation splits for R52 or R8.
        Args:
            device (Device): Device to use to store the dataset.
            r8 (bool, optional): If true, it initializes R8 instead of R52. Defaults to False.
            val_size (float, optional): Proportion of training documents to include in the validation set.
        Returns:
            train_split (Dataset): Training split.
            test_split (Dataset): Test split.
            val_split (Dataset): Validation split.
        """
        super(Reuters, self).__init__(root, transform, pre_transform)
        self.device = device
        self.word_embeddings = vocab.GloVe(name='840B', dim=300)
        self.doc_embeddings = RobertaModel.from_pretrained('roberta-base')
        self.doc_embeddings.eval()
        self.mode = mode
        self.tokenizer = tokenizer

        print('Prepare Reuters dataset')
        (train_docs, test_docs, val_docs), classes = self.prepare_reuters(r8, val_size)
        all_docs = train_docs + test_docs + val_docs
        corpus = [[word.lower() for word in reuters.words(doc)] for doc in all_docs]

        print('Compute tf.idf')
        tf_idf, words = tf_idf_mtx(corpus)

        print('Compute PMI scores')
        pmi = PMI(corpus)

        # Index to node name mapping
        self.iton = list(all_docs + words)
        # Node name to index mapping
        self.ntoi = {self.iton[i]: i for i in range(len(self.iton))}

        # Edge index and values for dataset
        print('Generate edges')
        edge_index, edge_attr = self.generate_edges(len(all_docs), tf_idf, pmi)

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
        doc_feats, word_feats = self.generate_features(words, len(all_docs), tokenizer)
        # node_feats = torch.eye(len(self.iton), device=self.device).float()
        #node_feats = torch.rand(size=(len(self.iton), 100), device=self.device).float()
        print('Documents features mtx is {} GBs in size'.format(doc_feats.nelement() * doc_feats.element_size() * 1e-9))
        print('Words features mtx is {} GBs in size'.format(word_feats.nelement() * word_feats.element_size() * 1e-9))

        # Create pytorch geometric format data
        padded_word_feats = torch.cat((word_feats, torch.zeros((word_feats.shape[0],doc_feats.shape[1]-word_feats.shape[1]))),dim=1)
        self.data = Data(x=torch.cat((doc_feats, padded_word_feats), dim=0), edge_index=edge_index, edge_attr=edge_attr, y=ntol)
        self.data.num_docs = doc_feats.shape[0]
        self.data.train_mask = train_mask
        self.data.val_mask = val_mask
        self.data.test_mask = test_mask

    def generate_features(self, words, num_docs, tokenizer):
        num_word_nodes = len(words)
        features_docs = torch.zeros((num_docs, 768))
        features_words = torch.zeros((num_word_nodes, 300))
        unk_embedding = torch.zeros((1, 300))

        for i in tqdm(range(num_docs)):
            doc_words = reuters.words(self.iton[i])
            sentence = ' '.join(doc_words)
            # encodings = tokenizer.encode(sentence, return_tensors='pt')
            # encodings = tokenizer(doc_words, truncation=True, padding=False)
            encodings = tokenizer([sentence], truncation=True, padding=False)['input_ids']
            encodings = torch.tensor(encodings, dtype=torch.long)
            # doc_tensor = encodings.convert_to_tensors
            embed_doc = self.doc_embeddings(encodings)[1]
            features_docs[i] = embed_doc
        
        for j, word in tqdm(enumerate(words)):
            word_idx = self.word_embeddings.stoi.get(word)
            if word_idx is None:
                embed_word = unk_embedding
            else:
                embed_word = self.word_embeddings.vectors[word_idx]
            features_words[j] = embed_word
        return features_docs, features_words
            

    def generate_edges(self, num_docs, tf_idf, pmi):
        """Generates edge list and weights based on tf.idf and PMI.

        Args:
            num_docs (int): Number of all documents in the dataset
            tf_idf (SparseMatrix): sklearn Sparse matrix object containing tf.idf values
            pmi (dict): Dictonary of word pairs and corresponding PMI scores

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
        for (word_i, word_j), score in pmi.items():
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

        # For testing with only a few docs:
        return (train_docs[:10], test_docs[:5], val_docs[:5]), unique_classes

        return (train_docs, test_docs, val_docs), unique_classes
    
    def len(self):
        length = len(self.data)
        return 1
    
    def get(self, idx):
        return self.data


class R52Graph(Reuters):
    """
    Wrapper for the R52 dataset.
    """
    def __init__(self, device, val_size=0.1):
        super().__init__(r8=False, device=device, val_size=val_size)


class R8Graph(Reuters):
    """
    Wrapper for the R8 dataset.
    """
    def __init__(self, device, tokenizer=None, val_size=0.1, root=None):
        super().__init__(root=root, r8=True, tokenizer=tokenizer, device=device, val_size=val_size)