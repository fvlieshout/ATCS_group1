import torch
from data_prep.dataset import GraphDataset
from datasets import load_dataset
from nltk.tokenize.regexp import WordPunctTokenizer
from torch_geometric.data import Data


class AGNewsGraph(GraphDataset):
    def __init__(self, device, val_size=0.1, n_train_docs=None):
        """
        Creates the train, test, and validation splits for AGNews.
        Args:
            device (Device): Device to use to store the dataset.
            val_size (float, optional): Proportion of training documents to include in the validation set.
            n_train_docs (int, optional): Number of documents to use from the training set. If None, include all.
        Returns:
            train_split (Dataset): Training split.
            test_split (Dataset): Test split.
            val_split (Dataset): Validation split.
        """
        super(GraphDataset, self).__init__(device, n_train_docs)

        print('Prepare AGNews dataset')
        docs, labels, classes = self.prepare_agnews(val_size, n_train_docs)

        train_labels, test_labels, val_labels = labels
        self.all_labels = train_labels + test_labels + val_labels

        self.tokenizer = WordPunctTokenizer()  # TODO: look into a better one?

        super(GraphDataset, self).initialize_data(docs, classes)

    @staticmethod
    def prepare_agnews(val_size=0.1, n_train_docs=None):
        """
        Return the training, validation, and tests splits along with the classes of AGNews dataset.
        Args:
            val_size (float, optional): Proportion of training documents to include in the validation set.
            n_train_docs (int, optional): Number of documents to use from the training set. If None, include all.
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

        if n_train_docs is not None:
            # For testing with only a few docs:
            n_test_val_docs = int(val_size * n_train_docs)
            docs = (list(train_texts[:n_train_docs]), list(test_texts[:n_test_val_docs]),
                    list(val_texts[:n_test_val_docs]))
            labels = (list(train_labels[:n_train_docs]), list(test_labels[:n_test_val_docs]),
                      list(val_labels[:n_test_val_docs]))
        else:
            docs = (list(train_texts), list(test_texts), list(val_texts))
            labels = (list(train_labels), list(test_labels), list(val_labels))

        return docs, labels, unique_classes

    def get_label_node_mapping(self):
        return self.all_labels + [-1] * self.n_words

    def pre_process_words(self, all_docs):
        return [self.tokenizer.tokenize(sentence) for sentence in all_docs]

    @property
    def num_classes(self):
        return len(self.itol)

    def __getitem__(self, idx):
        return self.data

    def __len__(self):
        return 1
