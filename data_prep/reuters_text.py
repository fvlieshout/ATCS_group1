#!/usr/bin/env python3
import random
from collections import defaultdict

import nltk

nltk.download('reuters')
from nltk.corpus import reuters

import torch
from data_prep.dataset import TextDataset
from torch.utils.data import Dataset

class Reuters(Dataset):
    def __init__(self, encodings, labels, classes):
        self.encodings = encodings
        self.labels = labels
        self.classes = classes


    @classmethod
    def splits(cls, tokenizer, r8=False, val_size=0.1):
        """
        Creates the train, test, and validation splits for R52 or R8.

        Args:
            tokenizer (Tokenizer): Hugging Face tokenizer to encode the 3 dataset splits.
            r8 (bool, optional): If true, it initializes R8 instead of R52. Defaults to False.
            val_size (float, optional): Proportion of training documents to include in the validation set.

        Returns:
            train_split (Dataset): Training split.
            test_split (Dataset): Test split.
            val_split (Dataset): Validation split.
        """
        (train_docs, test_docs, val_docs), unique_cls = cls.prepare_reuters(r8, val_size)

        train_split = cls.get_split(tokenizer, train_docs, unique_cls)
        test_split = cls.get_split(tokenizer, test_docs, unique_cls)
        val_split = cls.get_split(tokenizer, val_docs, unique_cls)

        return train_split, test_split, val_split

    @classmethod
    def get_split(cls, tokenizer, docs, unique_cls):
        texts, labels = cls._prepare_split(docs, unique_cls)
        encodings = tokenizer(texts, truncation=True, padding=True)
        print("HELLO", cls)
        return cls(encodings, labels, unique_cls)

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

    @staticmethod
    def _prepare_split(docs, classes):
        """
        Extract the text and labels of each documents.

        Args:
            docs (List): List of document names from which the texts and labels will be extracted.
            classes (List): List of unique class names sorted in alphabetical order.

        Returns:
            texts (List): List of Strings containing the text of the documents.
            labels (List): List of int containing the labels of the documents.
        """
        texts = []
        labels = []
        for doc in docs:
            text = ' '.join(reuters.words(doc))
            clz = reuters.categories(doc)[0]
            texts.append(text)
            labels.append(classes.index(clz))

        return texts, labels
    
    @property
    def num_classes(self):
        return len(self.classes)

    def __getitem__(self, idx):
        # assumes that the encodings were created using a HuggingFace tokenizer
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class R52(Reuters):
    """
    Wrapper for the R52 dataset.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def splits(cls, *args, **kwargs):
        return super().splits(r8=False, *args, **kwargs)


class R8(Reuters):
    """
    Wrapper for the R8 dataset.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def splits(cls, *args, **kwargs):
        return super().splits(r8=True, *args, **kwargs)


if __name__ == "__main__":
    from transformers import RobertaTokenizer
    from torch.utils.data import DataLoader

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    train_set, test_set, val_set = R52.splits(tokenizer)
    print("train size=", len(train_set))
    print("test size=", len(test_set))
    print("val size=", len(val_set))

    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)

    for data_batch in train_loader:
        input_ids = data_batch["input_ids"]
        attention_mask = data_batch["attention_mask"]
        d_labels = data_batch["labels"]
        print(input_ids)
        print(attention_mask)
        print(d_labels)
        break
