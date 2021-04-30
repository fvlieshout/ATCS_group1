#!/usr/bin/env python3
from collections import defaultdict
import random

import nltk
nltk.download('reuters')
from nltk.corpus import reuters
nltk.download('punkt')
from nltk import word_tokenize

from torchtext.data import Dataset, Example


class Reuters(Dataset):
    def __init__(self, docs, fields, **kwargs):
        """Initializes the Reuters dataset with the given documents and fields.

        Args:
            docs (list): List of documents to use.
            fields (list): List of fields in a format such that "fromlist" can be called.
        """
        examples = []
        for doc in docs:
            example = [doc, ' '.join(reuters.words(doc)), reuters.categories(doc)[0]]
            examples.append(Example.fromlist(example, fields))

        super().__init__(examples, fields, **kwargs)
        
    @classmethod
    def splits(cls, ID, TEXT, LABEL, r8=False, val_size=0.1):
        """Creates the train and test splits for R52 or R8.

        Args:
            ID (Field): Id field.
            TEXT (Field): Text field.
            LABEL (Field): Label field.
            r8 (bool, optional): If true, it initializes R8 instead of R52. Defaults to False.
            val_size (float, optional): Proportion of training documents to include in the validation set.

        Returns:
            train_split (Dataset): Training split.
            test_split (Dataset): Test split.
            val_split (Dataset): Validation split.
        """        
        train_docs, test_docs = cls.prepare_reuters(r8)
        fields = [('id', ID), ('text', TEXT), ('label', LABEL)]
        
        # Select the validation documents out of the training documents
        val_size = int(len(train_docs) * val_size)
        random.shuffle(train_docs)
        val_docs = train_docs[:val_size]
        train_docs = train_docs[val_size:]
        
        train_split = cls(train_docs, fields)
        test_split = cls(test_docs, fields)
        val_split = cls(val_docs, fields)
        return train_split, test_split, val_split

    @staticmethod
    def prepare_reuters(r8=False):
        """Filters out all documents which have more or less than 1 class. Then filters out all classes which have no remaining documents.

        Args:
            r8 (bool, optional): R8 is constructed by taking only the top 10 (original) classes. Defaults to False.

        Returns:
            train_docs (list): List of training documents.
            test_docs (list): List of test documents.
        """    
        # Filter out docs which don't have exactly 1 class
        data = defaultdict(lambda: {'train': [], 'test': []})
        for doc in reuters.fileids():
            # print("reuter field=", doc)
            if len(reuters.categories(doc)) == 1:
                if doc.startswith('training'):
                    data[reuters.categories(doc)[0]]['train'].append(doc)
                elif doc.startswith('test'):
                    data[reuters.categories(doc)[0]]['test'].append(doc)
                else:
                    print(doc)

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
        
        return train_docs, test_docs


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