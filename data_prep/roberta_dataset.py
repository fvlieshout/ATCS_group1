from data_prep.dataset import Dataset
from torch.utils.data import DataLoader
from transformers.data.data_collator import default_data_collator
import torch


class RobertaDataset(Dataset):
    """
    Text Dataset used by the Roberta model.
    """

    def __init__(self, data):
        super().__init__()

        self._labels = data.labels
        self._encodings = self._tokenizer(data.texts, truncation=True, padding=True)

    def as_dataloader(self, b_size, shuffle=False):
        return DataLoader(self, batch_size=b_size, num_workers=24, shuffle=shuffle, collate_fn=self.get_collate_fn())

    def get_collate_fn(self):
        """
        Return a function (collate_fn) to be used to preprocess a batch in the Dataloader.
        """

        return default_data_collator

        # if not isinstance(self._data, AGNewsData):
        #     return default_data_collator
        # else:
        #     def collate_fn(batch):
        #         texts = [data["text"] for data in batch]
        #         labels = [data["label"] for data in batch]
        #         encodings = self._tokenizer(texts, truncation=True, padding=True)
        #
        #         items = {key: torch.tensor(val) for key, val in encodings.items()}
        #         items["labels"] = torch.tensor(labels)
        #
        #         return items
        #
        #     return collate_fn

    def labels(self):
        """
        Return the labels of data points.
        """
        return self._labels

    def __getitem__(self, idx):
        # if isinstance(self._data, R8Data):

        # assumes that the encodings were created using a HuggingFace tokenizer
        item = {key: torch.tensor(val[idx]) for key, val in self._encodings.items()}
        item["labels"] = torch.tensor(self._labels[idx])
        return item

        # elif isinstance(self._data, AGNewsData):
        #     item = {
        #         "text": self.texts[idx],
        #         "label": self.labels[idx]
        #     }
        #     return item

    def __len__(self):
        return len(self._labels)
