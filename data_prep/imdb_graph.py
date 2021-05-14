from data_prep.dataset import HuggingFaceGraph

class IMDbGraph(HuggingFaceGraph):
    def __init__(self, device, val_size=0.1, n_train_docs=None):
        """
        Creates the train, test, and validation splits for IMDb.
        Args:
            device (Device): Device to use to store the dataset.
            val_size (float, optional): Proportion of training documents to include in the validation set.
            n_train_docs (int, optional): Number of documents to use from the training set. If None, include all.
        Returns:
            train_split (Dataset): Training split.
            test_split (Dataset): Test split.
            val_split (Dataset): Validation split.
        """
        super(IMDbGraph, self).__init__(device, val_size=val_size, n_train_docs=n_train_docs, dataset_name="imdb")
