import abc

import torch.utils.data as data
from transformers import RobertaTokenizerFast
import torch


class Dataset(data.Dataset, metaclass=abc.ABCMeta):
    """
    Parent class for all datasets.
    Require to implement a num_classes property and a labels method.
    """

    def __init__(self):
        super().__init__()

        self._tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

        # TODO: maybe move to GraphDataset (if not needed for roberta)
        self._device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    @property
    @abc.abstractmethod
    def num_classes(self):
        """
        Return the number of unique classes in the dataset.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def labels(self):
        """
        Return the labels of datapoints.
        """
        raise NotImplementedError
