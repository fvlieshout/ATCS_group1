import abc

class Data(metaclass=abc.ABCMeta):
    """
    Parent class for all datasets.
    Require to implement a train_data, test_data, val_data, and num_classes properties.
    """
    @property
    @abc.abstractmethod
    def train_data(self):
        """
        Get the training data as a tuple containing 2 elements:
            texts (List): List of training document contents as string.
            labels (List): List of training label ids as integer.
        """
        raise NotImplementedError
    
    @property
    @abc.abstractmethod
    def test_data(self):
        """
        Get the test data as a tuple containing 2 elements:
            texts (List): List of test document contents as string.
            labels (List): List of test label ids as integer.
        """
        raise NotImplementedError
    
    @property
    @abc.abstractmethod
    def val_data(self):
        """
        Get the val data as a tuple containing 2 elements:
            texts (List): List of val document contents as string.
            labels (List): List of val label ids as integer.
        """
        raise NotImplementedError
    
    @property
    @abc.abstractmethod
    def num_classes(self):
        """
        Get the number of unique classes in the dataset.
        """
        raise NotImplementedError
