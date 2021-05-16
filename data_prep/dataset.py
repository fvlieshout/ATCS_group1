import abc

import torch
import torch.utils.data as data
from transformers import RobertaTokenizerFast


class Dataset(data.Dataset, metaclass=abc.ABCMeta):
    """
    Parent class for all datasets.
    Require to implement a num_classes property and a labels method.
    """

    def __init__(self):
        super().__init__()

        self._tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    @abc.abstractmethod
    def labels(self):
        """
        Return the labels of datapoints.
        """
        raise NotImplementedError
