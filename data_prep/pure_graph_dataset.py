from data_prep.graph_dataset import GraphDataset


class PureGraphDataset(GraphDataset):
    """
    Text Dataset used by the pure graph model.
    """
    def __init__(self, *args, **kwargs):
        # class super init
        # append features to data object
        raise NotImplementedError

    def preprocess(self):
        """
        Preprocesses the corpus.
        """
        # Run Roberta tokenizer on corpus
        raise NotImplementedError

    def generate_features(self):
        """
        Generates node features.
        """
        raise NotImplementedError
