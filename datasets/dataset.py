import abc
import torch.utils.data as data

class Dataset(abc.ABCMeta):
    """
    Parent class for all datasets.
    Require to implement a num_classes property.

    """
    @abc.abstractmethod
    @property
    def num_classes(self):
        """
        Return the number of unique classes in the dataset.
        """
        raise NotImplementedError


class TextDataset(Dataset, data.Dataset):
    """
    Parent class for text datasets.
    Require to implement get_collate_fn.
    """
    @abc.abstractmethod
    def get_collate_fn(self):
        """
        Return a fonction (collate_fn) to be used to preprocess a batch in the Dataloader.
        """
        raise NotImplementedError

