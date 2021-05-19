import torch
from data_prep.dataset import Dataset
from torch.utils.data import DataLoader
from transformers.data.data_collator import default_data_collator


class RobertaDataset(Dataset):
    """
    Text Dataset used by the Roberta model.
    """

    def __init__(self, data):
        super().__init__()

        self._texts, self._labels = data

    def as_dataloader(self, b_size, shuffle=False):
        return DataLoader(self, batch_size=b_size, num_workers=24, shuffle=shuffle, collate_fn=self.get_collate_fn())

    def get_collate_fn(self):
        """
        Return a function (collate_fn) to be used to preprocess a batch in the Dataloader.
        We need to tokenize here and not when instantiating, because of memory issues.
        """

        def collate_fn(batch):
            texts = [data["text"] for data in batch]
            labels = [data["label"] for data in batch]
            encodings = self._tokenizer(texts, truncation=True, padding=True)
            items = {key: torch.tensor(val) for key, val in encodings.items()}
            items["labels"] = torch.tensor(labels)
            return items

        return collate_fn

    @staticmethod
    def tokenize(tokenizer, texts):
        encodings = tokenizer(texts, truncation=True, padding=True)
        return

    def labels(self):
        """
        Return the labels of data points.
        """
        return self._labels

    def __getitem__(self, idx):
        return {
            "text": self._texts[idx],
            "label": self._labels[idx]
        }

    def __len__(self):
        return len(self._labels)
