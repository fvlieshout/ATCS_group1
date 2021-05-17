from data_prep.dataset import Dataset


class RobertaDataset(Dataset):
    """
    Text Dataset used by the Roberta model.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def get_collate_fn(self):
        """
        Return a function (collate_fn) to be used to preprocess a batch in the Dataloader.
        """
        raise NotImplementedError

    @property
    def num_classes(self):
        """
        Return the number of unique classes in the dataset.
        """
        raise NotImplementedError

    def labels(self):
        """
        Return the labels of datapoints.
        """
        raise NotImplementedError
