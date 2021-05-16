from data_prep.agnews_data import *
from data_prep.graph_dataset import *
from data_prep.imdb_data import *
from data_prep.pure_graph_dataset import *
from data_prep.reuters_data import *
from data_prep.roberta_graph_dataset import *
from data_prep.roberta_dataset import *


def get_dataloaders(model, b_size, data_name):
    corpus = get_data(data_name)
    additional_params = {}

    if model == 'roberta':
        train_dataloader = RobertaDataset(corpus.train).as_dataloader(b_size, shuffle=True)
        test_dataloader = RobertaDataset(corpus.test).as_dataloader(b_size)
        val_dataloader = RobertaDataset(corpus.val).as_dataloader(b_size)

        additional_params['num_classes'] = corpus.num_classes

        return train_dataloader, test_dataloader, val_dataloader, additional_params

    if model == 'pure_gnn':
        dataset = PureGraphDataset(corpus)
    elif model == 'roberta_gnn':
        dataset = RobertaGraphDataset(corpus)
    else:
        raise ValueError("Model type '%s' is not supported." % model)

    if isinstance(dataset, GraphDataset):
        additional_params['num_classes'] = corpus.num_classes
        additional_params['gnn_output_dim'] = dataset.num_nodes
        train_loader = val_loader = test_loader = dataset.as_dataloader()
        return train_loader, val_loader, test_loader, additional_params


def get_data(data_name):
    if data_name in ['R8', 'R52']:
        return ReutersData(r8=data_name == 'R8')
    elif data_name == 'AGNews':
        return AGNewsText()
    elif data_name == 'IMDb':
        return IMDbData()
    else:
        raise ValueError("Data with name '%s' is not supported." % data_name)
