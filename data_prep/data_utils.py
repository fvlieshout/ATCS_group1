from data_prep.agnews_data import *
from data_prep.glove_graph_dataset import *
from data_prep.graph_dataset import *
from data_prep.imdb_data import *
from data_prep.reuters_data import *
from data_prep.roberta_dataset import *
from data_prep.roberta_graph_dataset import *

SUPPORTED_DATASETS = ['R8', 'R52', 'AGNews', 'IMDb']


def get_dataloaders(model, b_size, data_name, roberta_model=None):
    """
    Initializes train, text and validation dataloaders for either roberta or graph models.
    Args:
        model (str): The name of the model which will be used.
        b_size (int): Batch size to be used in the data loader.
        data_name (str): Name of the data corpus which should be used.
        roberta_model (str): Checkpoint path for a pretrained/fine-tuned Roberta model.
    Returns:
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        test_loader (DataLoader): Test data loader.
        additional_params (dict): Additional parameters needed for instantiation of the actual model later.
    Raises:
        Exception: if the model is not in ['roberta', 'glove_gnn', 'roberta_pretrained_gnn', 'roberta_finetuned_gnn']
    """
    corpus = get_data(data_name)
    additional_params = {}

    if model == 'roberta':
        train_loader = RobertaDataset(corpus.train).as_dataloader(b_size, shuffle=True)
        test_loader = RobertaDataset(corpus.test).as_dataloader(b_size)
        val_loader = RobertaDataset(corpus.val).as_dataloader(b_size)

        additional_params['num_classes'] = corpus.num_classes

        return train_loader, val_loader, test_loader, additional_params

    if model == 'glove_gnn':
        dataset = GloveGraphDataset(corpus)
        additional_params['doc_dim'] = dataset.doc_dim
        additional_params['word_dim'] = dataset.word_dim
    elif model in ['roberta_pretrained_gnn', 'roberta_finetuned_gnn']:
        dataset = RobertaGraphDataset(corpus, roberta_model)
    else:
        raise ValueError("Model type '%s' is not supported." % model)

    if isinstance(dataset, GraphDataset):
        additional_params['num_classes'] = corpus.num_classes
        additional_params['gnn_output_dim'] = dataset.num_nodes
        train_loader = val_loader = test_loader = dataset.as_dataloader()
        return train_loader, val_loader, test_loader, additional_params


def get_data(data_name, val_size=0.1):
    """
    Creates and returns the correct data object depending on data_name.
    Args:
        data_name (str): Name of the data corpus which should be used.
        val_size (float, optional): Proportion of training documents to include in the validation set.
    Raises:
        Exception: if the data_name is not in ['R8', 'R52', 'AGNews', 'IMDb'].
    """

    if data_name not in SUPPORTED_DATASETS:
        raise ValueError("Data with name '%s' is not supported." % data_name)

    if data_name == 'R8':
        return R8Data(val_size=val_size)
    elif data_name == 'R52':
        return R52Data(val_size=val_size)
    elif data_name == 'AGNews':
        return AGNewsData(val_size=val_size)
    elif data_name == 'IMDb':
        return IMDbData(val_size=val_size)
    else:
        raise ValueError("Data with name '%s' is not supported." % data_name)
