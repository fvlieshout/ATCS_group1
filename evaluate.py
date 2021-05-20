import argparse

from train import train


def evaluate(model_name, seed, epochs, patience, b_size, l_rate_enc, l_rate_cl, w_decay_enc, w_decay_cl, warmup,
             cf_hidden_dim, data_name, checkpoint, gnn_layer_name, transfer):
    return train(model_name, seed, epochs, patience, b_size, l_rate_enc, l_rate_cl, w_decay_enc, w_decay_cl, warmup,
                 cf_hidden_dim, data_name, checkpoint, gnn_layer_name, transfer, h_search=False, eval=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # TRAINING PARAMETERS

    parser.add_argument('--epochs', dest='epochs', type=int, default=50)
    parser.add_argument('--patience', dest='patience', type=int, default=10)
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=64)
    parser.add_argument('--lr-enc', dest='l_rate_enc', type=float, default=0.01,
                        help="Encoder learning rate.")
    parser.add_argument('--lr-cl', dest='l_rate_cl', type=float, default=-1,
                        help="Classifier learning rate.")
    parser.add_argument("--w-decay-enc", dest='w_decay_enc', type=float, default=2e-3,
                        help="Encoder weight decay for L2 regularization of optimizer AdamW")
    parser.add_argument("--w-decay-cl", dest='w_decay_cl', type=float, default=-1,
                        help="Classifier weight decay for L2 regularization of optimizer AdamW")
    parser.add_argument("--warmup", dest='warmup', type=int, default=500,
                        help="Number of steps for which we do learning rate warmup.")

    # CONFIGURATION

    parser.add_argument('--dataset', dest='dataset', default='R8', choices=SUPPORTED_DATASETS,
                        help='Select the dataset you want to use.')
    parser.add_argument('--model', dest='model', default='roberta_pretrained_gnn', choices=SUPPORTED_MODELS,
                        help='Select the model you want to use.')
    parser.add_argument('--gnn-layer-name', dest='gnn_layer_name', default='GCNConv', choices=SUPPORTED_GNN_LAYERS,
                        help='Select the GNN layer you want to use.')
    parser.add_argument('--seed', dest='seed', type=int, default=1234)
    parser.add_argument('--cf-hidden-dim', dest='cf_hidden_dim', type=int, default=512)
    parser.add_argument('--checkpoint', default=None, type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: None)')
    parser.add_argument('--transfer', dest='transfer', action='store_true', help='Transfer the model to new dataset.')

    params = vars(parser.parse_args())

    evaluate(
        model_name=params['model'],
        seed=params['seed'],
        epochs=params['epochs'],
        patience=params['patience'],
        b_size=params["batch_size"],
        l_rate_enc=params["l_rate_enc"],
        l_rate_cl=params["l_rate_cl"],
        w_decay_enc=params["w_decay_enc"],
        w_decay_cl=params["w_decay_cl"],
        warmup=params["warmup"],
        cf_hidden_dim=params["cf_hidden_dim"],
        data_name=params["dataset"],
        checkpoint=params["checkpoint"],
        gnn_layer_name=params["gnn_layer_name"],
        transfer=params["transfer"],
    )
