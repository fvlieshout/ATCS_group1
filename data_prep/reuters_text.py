#!/usr/bin/env python3
import nltk

nltk.download('reuters')
from nltk.corpus import reuters

import torch
from data_prep.dataset import TextDataset, Reuters
from transformers.data.data_collator import default_data_collator


class ReutersText(Reuters, TextDataset):
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
            train_split (ReutersText): Training split.
            test_split (ReutersText): Test split.
            val_split (ReutersText): Validation split.
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
        return cls(encodings, labels, unique_cls)

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

    def get_collate_fn(self):
        """
        No specific collate function required.
        """
        return default_data_collator

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


class R52Text(ReutersText):
    """
    Wrapper for the R52 dataset.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def splits(cls, *args, **kwargs):
        return super().splits(r8=False, *args, **kwargs)


class R8Text(ReutersText):
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

    train_set, test_set, val_set = R52Text.splits(tokenizer)
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
