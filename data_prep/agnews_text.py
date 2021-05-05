#!/usr/bin/env python3
import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class AGNews(Dataset):
    def __init__(self, encodings, labels, classes):
        self.encodings = encodings
        self.labels = labels
        self.classes = classes
    
    @classmethod
    def splits(cls, tokenizer, val_size=0.1):
        """
        Creates the train, test, and validation splits for AGnews.

        Args:
            tokenizer (Tokenizer): Hugging Face tokenizer to encode the 3 dataset splits.
            val_size (float, optional): Proportion of training sample to include in the validation set.

        Returns:
            train_split (Dataset): Training split.
            test_split (Dataset): Test split.
            val_split (Dataset): Validation split.
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
        encodings = tokenizer(texts, truncation=True, padding=True)
        unique_cls = dataset.features["label"].names
        return cls(encodings, labels, unique_cls)
    
    @staticmethod
    def _prepare_split(dataset):
        texts = []
        labels = []
        for data in dataset:
            texts.append(data["text"])
            labels.append(data["label"])

        return texts, labels
    
    def __getitem__(self, idx):
        # assumes that the encodings were created using a HuggingFace tokenizer
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
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

    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)

    for data_batch in train_loader:
        input_ids = data_batch["input_ids"]
        attention_mask = data_batch["attention_mask"]
        d_labels = data_batch["labels"]
        print(input_ids)
        print(attention_mask)
        print(d_labels)
        break


