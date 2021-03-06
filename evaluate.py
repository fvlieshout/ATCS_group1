import argparse

from train import *


def evaluate(model_name, seed, b_size, data_name, checkpoint, transfer):
    """
    Runs evaluation on training and validation set for a model which is loaded from a checkpoint.

    Args:
        model_name (str) - Name of the model which should be evaluated.
        seed (str) - The seed used for this.
        b_size (str) - Batch size.
        data_name (str) - Name of the data corpus which should be used.
        checkpoint (str) - Path to the checkpoint file.
        transfer (bool) - If doing transfer learning or not.
    Returns:
        test_accuracy (float) - The achieved test accuracy.
        val_accuracy (float) - The achieved validation accuracy.
    """
    return train(model_name, seed, -1, -1, b_size, -1, -1, -1, -1, -1, 1, data_name, checkpoint, None, None, transfer,
                 h_search=False, eval=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # EVALUATION PARAMETERS

    parser.add_argument('--batch-size', dest='batch_size', type=int, default=64)

    # CONFIGURATION

    parser.add_argument('--dataset', dest='dataset', default='R8', choices=SUPPORTED_DATASETS,
                        help='Select the dataset you want to use.')
    parser.add_argument('--model', dest='model', default='roberta_pretrained_gnn', choices=SUPPORTED_MODELS,
                        help='Select the model you want to use.')
    parser.add_argument('--seed', dest='seed', type=int, default=1234)
    parser.add_argument('--checkpoint', default=None, type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: None)')
    parser.add_argument('--transfer', dest='transfer', action='store_true', help='Transfer the model to new dataset.')

    params = vars(parser.parse_args())

    evaluate(
        model_name=params['model'],
        seed=params['seed'],
        b_size=params["batch_size"],
        data_name=params["dataset"],
        checkpoint=params["checkpoint"],
        transfer=params["transfer"],
    )
