import abc
import random
from collections import defaultdict
import torch.utils.data as data
from nltk.corpus import reuters


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
