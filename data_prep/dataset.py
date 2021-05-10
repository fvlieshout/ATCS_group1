import abc
import random
from collections import defaultdict

import torch
import torch.utils.data as data
from nltk.corpus import reuters
from datasets import load_dataset


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

class HuggingFaceDataset(TextDataset):
    """
    Parent class HuggingFace text datasets.
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