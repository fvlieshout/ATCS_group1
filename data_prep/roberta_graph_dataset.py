from data_prep.graph_dataset import GraphDataset


class RobertaGraphDataset(GraphDataset):
    """
    Text Dataset used by the Roberta graph model.
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
