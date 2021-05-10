import argparse
import os
import time

import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
import torch
from data_prep.reuters_text import R8Text, R52Text
from models.model import ClassifierModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast

# disable parallelism for hugging face to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# more specific cuda errors
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

LOG_PATH = "./logs/"

SUPPORTED_MODELS = ['roberta']
SUPPORTED_DATASETS = ['R8Text', 'R52Text']


def train(model_name, seed, epochs, patience, b_size, l_rate, l_decay, minimum_lr, cf_hidden_dim,
          dataset_name='R8Text'):
    os.makedirs(LOG_PATH, exist_ok=True)

    print(f'Configuration:\n model_name: {model_name} \n max epochs: {epochs}\n patience: {patience}'
          f'\n seed: {seed}\n batch_size: {b_size}\n l_rate: {l_rate}\n l_decay: {l_decay}\n '
          f'cf_hidden_dim: {cf_hidden_dim}\n dataset_name: {dataset_name}\n')

    pl.seed_everything(seed)

    # the data preprocessing per model

    dataset = get_dataset(dataset_name)

    if model_name == 'roberta':

        # Prepare the data

        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

        train_dataset, test_dataset, val_dataset = dataset.splits(tokenizer, val_size=0.1)

        train_dataloader = data_loader(b_size, train_dataset, shuffle=True)
        test_dataloader = data_loader(b_size, test_dataset)
        val_dataloader = data_loader(b_size, val_dataset)

    else:
        raise ValueError("Model type '%s' is not supported." % model_name)

    model_params = {'model': model_name, "num_classes": train_dataset.num_classes, "cf_hid_dim": cf_hidden_dim}
    optimizer_hparams = {"lr": l_rate, "weight_decay": l_decay}

    model = ClassifierModule(model_params, optimizer_hparams)

    trainer = initialize_trainer(epochs, patience, minimum_lr, model_name, l_rate, l_decay)

    # Training
    print('Fitting model ..........\n')
    start = time.time()
    trainer.fit(model, train_dataloader, val_dataloader)

    end = time.time()
    elapsed = end - start
    print(f'\nRequired time for training: {int(elapsed / 60)} minutes.\n')

    # Testing
    # Load best checkpoint after training
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f'Best model path: {best_model_path}')

    model = model.load_from_checkpoint(best_model_path)
    test_acc, val_acc = evaluate(trainer, model, test_dataloader, val_dataloader)

    # We want to save the whole model, because we fine-tune anyways!

    return test_acc, val_acc


def data_loader(b_size, dataset, shuffle=False):
    return DataLoader(dataset, batch_size=b_size, num_workers=24, shuffle=shuffle, collate_fn=dataset.get_collate_fn())


def evaluate(trainer, model, test_dataloader, val_dataloader):
    """
    Tests a model on test and validation set.
    """

    print('Testing model on validation and test ..........\n')

    test_start = time.time()

    test_result = trainer.test(model, test_dataloaders=test_dataloader, verbose=False)[0]
    test_accuracy = test_result["test_accuracy"]

    val_result = trainer.test(model, test_dataloaders=val_dataloader, verbose=False)[0]
    val_accuracy = val_result["test_accuracy"] if "val_accuracy" not in val_result else val_result["val_accuracy"]

    test_end = time.time()
    test_elapsed = test_end - test_start

    print(f'\nRequired time for testing: {int(test_elapsed / 60)} minutes.\n')
    print(f'Test Results:\n test accuracy: {round(test_accuracy, 3)} ({test_accuracy})\n '
          f'validation accuracy: {round(val_accuracy, 3)} ({val_accuracy})'
          f'\n epochs: {trainer.current_epoch + 1}\n')

    return test_accuracy, val_accuracy


def initialize_trainer(epochs, patience, minimum_lr, model_name, l_rate, l_decay):
    model_checkpoint = cb.ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_accuracy")

    os.makedirs(LOG_PATH, exist_ok=True)

    version_str = f'patience={patience}_lr={l_rate}_ldec={l_decay}'
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


def get_dataset(dataset_name):
    if dataset_name == "R8Text":
        return R8Text
    elif dataset_name == "R52Text":
        return R52Text
    else:
        raise ValueError("Dataset '%s' is not supported." % dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # TRAINING PARAMETERS

    parser.add_argument('--epochs', dest='epochs', type=int, default=50)
    parser.add_argument('--patience', dest='patience', type=int, default=10)
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=64)
    parser.add_argument('--lr', dest='l_rate', type=float, default=1e-4)
    parser.add_argument("--min-lr", dest='minimum_lr', type=float, default=1e-5, help="Minimum Learning Rate")
    parser.add_argument("--lr-decay", dest='lr_decay', type=float, default=1e-3, help="Learning rate (weight) decay")

    # CONFIGURATION

    parser.add_argument('--dataset', dest='dataset', default='R8Text', choices=SUPPORTED_DATASETS,
                        help='Select the dataset you want to use.')
    parser.add_argument('--model', dest='model', default='roberta', choices=SUPPORTED_MODELS,
                        help='Select the model you want to use.')
    parser.add_argument('--seed', dest='seed', type=int, default=1234)
    parser.add_argument('--cf-hidden-dim', dest='cf_hidden_dim', type=int, default=512)

    params = vars(parser.parse_args())

    train(
        model_name=params['model'],
        seed=params['seed'],
        epochs=params['epochs'],
        patience=params['patience'],
        b_size=params["batch_size"],
        l_rate=params["l_rate"],
        l_decay=params["lr_decay"],
        minimum_lr=params["minimum_lr"],
        cf_hidden_dim=params["cf_hidden_dim"],
        dataset_name=params["dataset"]
    )
