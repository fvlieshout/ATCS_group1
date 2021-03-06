import abc

from datasets import load_dataset


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


class HuggingFaceData(Data):
    def __init__(self, dataset_name, val_size=0.1):
        splits, unique_cls = self.prepare_data(dataset_name, val_size=val_size)
        train_split, test_split, val_split = splits

        self.train = self._prepare_split(train_split)
        self.test = self._prepare_split(test_split)
        self.val = self._prepare_split(val_split)

        self.classes = unique_cls

    @staticmethod
    def prepare_data(dataset_name, val_size=0.1):
        """
        Download the dataset and create the train, test, and validation splits.
        Args:
            dataset_name (string): Name fo the HuggingFace dataset.
            val_size (float, optional): Proportion of training sample to include in the validation set.
        Returns:
            doc_splits (tuple): Tuple containing the 3 training, test, and validation datasets.
        """
        dataset = load_dataset(dataset_name)
        train_val_splits = dataset["train"].train_test_split(test_size=val_size)

        unique_cls = dataset["train"].features["label"].names
        return (train_val_splits["train"], dataset["test"], train_val_splits["test"]), unique_cls

    @staticmethod
    def _prepare_split(dataset):
        texts = [data["text"] for data in dataset]
        labels = [data["label"] for data in dataset]
        return texts, labels

    @property
    def train_data(self):
        """
        Get the training data as a tuple containing 2 elements:
            texts (List): List of training document contents as string.
            labels (List): List of training label ids as integer.
        """
        return self.train

    @property
    def test_data(self):
        """
        Get the test data as a tuple containing 2 elements:
            texts (List): List of test document contents as string.
            labels (List): List of test label ids as integer.
        """
        return self.test

    @property
    def val_data(self):
        """
        Get the val data as a tuple containing 2 elements:
            texts (List): List of val document contents as string.
            labels (List): List of val label ids as integer.
        """
        return self.val

    @property
    def num_classes(self):
        """
        Get the number of unique classes in the dataset.
        """
        return len(self.classes)