import nltk

nltk.download('reuters')
from nltk.corpus import reuters
from data_prep.dataset import RobertaGraphDataset, Reuters


class ReutersGraph(RobertaGraphDataset, Reuters):
    def __init__(self, device, r8=False, val_size=0.1, n_train_docs=None, tokenizer=None):
        """
        Creates the train, test, and validation splits for R52 or R8.
        Args:
            device (Device): Device to use to store the dataset.
            r8 (bool, optional): If true, it initializes R8 instead of R52. Defaults to False.
            val_size (float, optional): Proportion of training documents to include in the validation set.
            n_train_docs (int, optional): Number of documents to use from the training set. If None, include all.
        Returns:
            train_split (Dataset): Training split.
            test_split (Dataset): Test split.
            val_split (Dataset): Validation split.
        """
        super(ReutersGraph, self).__init__(device, n_train_docs, tokenizer)

        print('Prepare Reuters dataset')
        docs, classes = self.prepare_reuters(r8, val_size, n_train_docs)

        super(ReutersGraph, self).initialize_data(docs, classes)

    def get_label_node_mapping(self):
        return [self.loti[reuters.categories(node)[0]] if reuters.categories(node) else -1 for node in self.iton]

    # noinspection PyMethodMayBeStatic
    def pre_process_words(self, all_docs):
        return [[word.lower() for word in reuters.words(doc)] for doc in all_docs]

    @staticmethod
    def prepare_reuters(r8=False, val_size=0.1, n_train_docs=None):
        (train_docs, test_docs, val_docs), unique_classes = Reuters.prepare_reuters(r8=r8, val_size=val_size)

        if n_train_docs is not None:
            # For testing with only a few docs:
            n_test_val_docs = int(val_size * n_train_docs)
            return (train_docs[:n_train_docs], test_docs[:n_test_val_docs], val_docs[:n_test_val_docs]), unique_classes
        else:
            return (train_docs, test_docs, val_docs), unique_classes

    @property
    def num_classes(self):
        return len(self.itol)

    def __getitem__(self, idx):
        return self.data

    def __len__(self):
        return 1


class R52Graph(ReutersGraph):
    """
    Wrapper for the R52 dataset.
    """

    def __init__(self, device, val_size=0.1, n_train_docs=None):
        super().__init__(r8=False, device=device, val_size=val_size, n_train_docs=n_train_docs)


class R8Graph(ReutersGraph):
    """
    Wrapper for the R8 dataset.
    """

    def __init__(self, device, val_size=0.1, n_train_docs=None, tokenizer=None):
        super().__init__(r8=True, device=device, val_size=val_size, n_train_docs=n_train_docs, tokenizer=tokenizer)
