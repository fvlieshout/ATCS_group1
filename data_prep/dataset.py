import abc

import torch.utils.data as data


class Dataset(data.Dataset, metaclass=abc.ABCMeta):
    """
    Parent class for all datasets.
    Require to implement a num_classes and labels property.
    """

    @property
    @abc.abstractmethod
    def num_classes(self):
        """
        Return the number of unique classes in the dataset.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def labels(self):
        """
        Return the labels of datapoints.
        """
        raise NotImplementedError
