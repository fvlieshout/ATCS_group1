#!/usr/bin/env python3
import torch
from data_prep.dataset import TextDataset
from datasets import load_dataset

class AGNewsText(TextDataset):
    def __init__(self, texts, labels, classes, tokenizer):
        self.texts = texts
        self.labels = labels
        self.classes = classes
        self.tokenizer = tokenizer
    
    @classmethod
    def splits(cls, tokenizer, val_size=0.1):
        """
        Creates the train, test, and validation splits for AGnews.

        Args:
            tokenizer (Tokenizer): Hugging Face tokenizer to encode the 3 dataset splits.
            val_size (float, optional): Proportion of training sample to include in the validation set.

        Returns:
            train_split (TextDataset): Training split.
            test_split (TextDataset): Test split.
            val_split (TextDataset): Validation split.
        """

        dataset = load_dataset("ag_news")
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
        texts = [data["text"] for data in dataset]
        labels = [data["label"] for data in dataset]
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


if __name__ == "__main__":
    from transformers import RobertaTokenizer
    from torch.utils.data import DataLoader

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    train_set, test_set, val_set = AGNews.splits(tokenizer)
    print("train size=", len(train_set))
    print("test size=", len(test_set))
    print("val size=", len(val_set))

    train_loader = DataLoader(train_set, batch_size=2, shuffle=True, collate_fn=train_set.get_collate_fn())

    for data_batch in train_loader:
        print(data_batch)
        input_ids = data_batch["input_ids"]
        attention_mask = data_batch["attention_mask"]
        d_labels = data_batch["labels"]
        print(input_ids)
        print(attention_mask)
        print(d_labels)
        break


