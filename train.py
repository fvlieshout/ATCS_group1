import argparse
import os
import time

import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
import torch
from data_prep.data_utils import get_dataloaders
from models.document_classifier import DocumentClassifier
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# disable parallelism for hugging face to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# more specific cuda errors
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

LOG_PATH = "./logs/"

SUPPORTED_MODELS = ['roberta', 'glove_gnn', 'roberta_pretrained_gnn', 'roberta_finetuned_gnn']
SUPPORTED_GNN_LAYERS = ['GCNConv', 'GraphConv']
SUPPORTED_DATASETS = ['R8', 'R52', 'AGNews', 'IMDb']


def train(model_name, seed, epochs, patience, b_size, l_rate_enc, l_rate_cl, w_decay_enc, w_decay_cl, warmup,
          cf_hidden_dim, data_name, checkpoint, gnn_layer_name, transfer, h_search):
    os.makedirs(LOG_PATH, exist_ok=True)

    if model_name not in SUPPORTED_MODELS:
        raise ValueError("Model type '%s' is not supported." % model_name)

    print(
        f'Configuration:\n model_name: {model_name}\n data_name: {data_name}\n max epochs: {epochs}\n'
        f' patience: {patience}\n seed: {seed}\n batch_size: {b_size}\n l_rate_enc: {l_rate_enc}\n l_rate_cl: {l_rate_cl}\n warmup: {warmup}\n '
        f'weight_decay_enc: {w_decay_enc}\n weight_decay_cl: {w_decay_cl}\n cf_hidden_dim: {cf_hidden_dim}\n checkpoint: {checkpoint}\n'
        f' h_search: {h_search}\n GNN layer: {gnn_layer_name}\n')

    pl.seed_everything(seed)

    # the data preprocessing

    train_loader, val_loader, test_loader, add_params = get_dataloaders(model_name, b_size, data_name, checkpoint)

    optimizer_hparams = {"lr_enc": l_rate_enc,
                         "lr_cl": l_rate_cl,
                         "weight_decay_enc": w_decay_enc,
                         "weight_decay_cl": w_decay_cl,
                         "warmup": warmup,
                         "max_iters": len(train_loader) * epochs}

    model_params = {
        'model': model_name,
        'gnn_layer_name': gnn_layer_name,
        'cf_hid_dim': cf_hidden_dim,
        **add_params
    }

    trainer = initialize_trainer(epochs, patience, model_name, l_rate_enc, l_rate_cl, w_decay_enc, w_decay_cl, warmup,
                                 seed, data_name, transfer)
    model = DocumentClassifier(model_params, optimizer_hparams, checkpoint=resume, transfer=transfer, h_search=h_search)

    # Training
    print('Fitting model ..........\n')
    start = time.time()
    trainer.fit(model, train_loader, val_loader)

    end = time.time()
    elapsed = end - start
    print(f'\nRequired time for training: {int(elapsed / 60)} minutes.\n')

    # Testing
    # Load best checkpoint after training
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f'Best model path: {best_model_path}')

    model = model.load_from_checkpoint(best_model_path)
    test_acc, val_acc = evaluate(trainer, model, test_loader, val_loader)

    # We want to save the whole model, because we fine-tune anyways!

    return test_acc, val_acc


def evaluate(trainer, model, test_dataloader, val_dataloader):
    """
    Tests a model on test and validation set.
    """

    print('Testing model on validation and test ..........\n')

    test_start = time.time()

    model.test_val_mode = 'test'
    test_result = trainer.test(model, test_dataloaders=test_dataloader, verbose=False)[0]
    test_accuracy = test_result["test_accuracy"]

    model.test_val_mode = 'val'
    val_result = trainer.test(model, test_dataloaders=val_dataloader, verbose=False)[0]
    val_accuracy = val_result["test_accuracy"] if "val_accuracy" not in val_result else val_result["val_accuracy"]
    model.test_val_mode = 'test'

    test_end = time.time()
    test_elapsed = test_end - test_start

    print(f'\nRequired time for testing: {int(test_elapsed / 60)} minutes.\n')
    print(f'Test Results:\n test accuracy: {round(test_accuracy, 3)} ({test_accuracy})\n '
          f'validation accuracy: {round(val_accuracy, 3)} ({val_accuracy})'
          f'\n epochs: {trainer.current_epoch + 1}\n')

    return test_accuracy, val_accuracy


def initialize_trainer(epochs, patience, model_name, l_rate_enc, l_rate_cl, weight_decay_enc, weight_decay_cl, warmup,
                       seed, dataset, transfer):
    model_checkpoint = cb.ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_accuracy")

    os.makedirs(LOG_PATH, exist_ok=True)

    if transfer:
        model_name = f'{model_name}-transfer'

    version_str = f'dname={dataset}_seed={seed}_lr-enc={l_rate_enc}_lr-cl={l_rate_cl}_wdec-enc={weight_decay_enc}_wdec-cl={weight_decay_cl}_wsteps={warmup}'

    logger = TensorBoardLogger(LOG_PATH, name=model_name, version=version_str)

    early_stop_callback = EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.00,
        patience=patience,  # validation happens per default after each training epoch
        verbose=False,
        mode='max'
    )

    trainer = pl.Trainer(logger=logger,
                         checkpoint_callback=model_checkpoint,
                         gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs=epochs,
                         callbacks=[early_stop_callback],
                         progress_bar_refresh_rate=1)

    # Optional logging argument that we don't need
    trainer.logger._default_hp_metric = None

    return trainer


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
    parser.add_argument('--h-search', dest='h_search', action='store_true', default=False,
                        help='Flag for doing hyper parameter search (and freezing half of roberta layers) '
                             'or doing full fine tuning.')

    params = vars(parser.parse_args())

    train(
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
        h_search=params["h_search"],
    )
