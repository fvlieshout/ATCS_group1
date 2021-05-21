from data_prep.agnews_data import *
from data_prep.glove_graph_dataset import *
from data_prep.graph_dataset import *
from data_prep.imdb_data import *
from data_prep.reuters_data import *
from data_prep.roberta_dataset import *
from data_prep.roberta_graph_dataset import *


def get_dataloaders(model, b_size, data_name, roberta_model=None):
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
