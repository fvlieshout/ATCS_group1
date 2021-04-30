#!/usr/bin/env python3
from collections import defaultdict
import random

import nltk
nltk.download('reuters')
from nltk.corpus import reuters

from torch.utils.data import Dataset
import torch


class Reuters(Dataset):
    def __init__(self, encodings, labels, classes):
        self.encodings = encodings
        self.labels = labels
        self.classes = classes
        
    @classmethod
    def splits(cls, tokenizer, r8=False, val_size=0.1):
        """Creates the train, test, and validation splits for R52 or R8.

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
        
        # Maybe this should be in a function
        train_texts, train_labels = cls._prepare_split(train_docs, unique_cls)
        train_encodings = tokenizer(train_texts, truncation=True, padding=True)
        train_split = cls(train_encodings, train_labels, unique_cls)

        test_texts, test_labels = cls._prepare_split(test_docs, unique_cls)
        test_encodings = tokenizer(test_texts, truncation=True, padding=True)
        test_split = cls(test_encodings, test_labels, unique_cls)

        val_texts, val_labels = cls._prepare_split(val_docs, unique_cls)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True)
        val_split = cls(val_encodings, val_labels, unique_cls)

        return train_split, test_split, val_split
    
    @staticmethod
    def prepare_reuters(r8=False, val_size=0.1):
        """Filters out all documents which have more or less than 1 class. Then filters out all classes which have no remaining documents.

        Args:
            r8 (bool, optional): R8 is constructed by taking only the top 10 (original) classes. Defaults to False.
            val_size (float, optional): Proportion of training documents to include in the validation set.

        Returns:
            doc_splits (tupple): Tupple containing 3 List of training, test, and validation documents.
            unique_classes (List): List of Strings containing the class names sorted in alphabetical order.
        """    
        # Filter out docs which don't have exactly 1 class
        data = defaultdict(lambda: {'train': [], 'test': []})
        for doc in reuters.fileids():
            if len(reuters.categories(doc)) == 1:
                if doc.startswith('training'):
                    data[reuters.categories(doc)[0]]['train'].append(doc)
                if doc.startswith('test'):
                    data[reuters.categories(doc)[0]]['test'].append(doc)

        # Filter out classes which have no remaining docs
        for cls in reuters.categories():
            if len(data[cls]['train']) < 1 or len(data[cls]['test']) < 1:
                data.pop(cls, None)
        
        if r8:
            # Choose top 10 classes and then select the ones which still remain after filtering
            popular = sorted(reuters.categories(), key=lambda cls: len(reuters.fileids(cls)), reverse=True)[:10]
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
        """Extract the text and labels of each documents.

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
            cls = reuters.categories(doc)[0]
            texts.append(text)
            labels.append(classes.index(cls))
        
        return texts, labels
    
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

    trainset, testset, valset = R52.splits(tokenizer, val_size=0.1)
    print("train size=", len(trainset))
    print("test size=", len(testset))
    print("val size=", len(valset))

    trainloader = DataLoader(trainset, batch_size=2)

    for data in trainloader:
        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]
        labels = data["labels"]
        print(input_ids)
        print(attention_mask)
        print(labels)
        break

