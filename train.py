import argparse
import os
import time

import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
import torch
from data_prep.data_utils import get_dataloaders
from models.model import DocumentClassifier
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# disable parallelism for hugging face to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# more specific cuda errors
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

LOG_PATH = "./logs/"

SUPPORTED_MODELS = ['roberta', 'pure_gnn', 'roberta_gnn']
SUPPORTED_DATASETS = ['R8', 'R52', 'AGNews', 'IMDb']


def train(model_name, seed, epochs, patience, b_size, l_rate, w_decay, warmup, max_iters, cf_hidden_dim, data_name,
          resume):
    os.makedirs(LOG_PATH, exist_ok=True)

    if model_name not in SUPPORTED_MODELS:
        raise ValueError("Model type '%s' is not supported." % model_name)

    print(f'Configuration:\n model_name: {model_name}\n data: {data_name}\n max epochs: {epochs}\n patience: {patience}'
          f'\n seed: {seed}\n batch_size: {b_size}\n l_rate: {l_rate}\n warmup: {warmup}\n '
          f'weight_decay: {w_decay}\n cf_hidden_dim: {cf_hidden_dim}\n resume checkpoint: {resume}\n')

    pl.seed_everything(seed)

    # the data preprocessing

    train_loader, test_loader, val_loader, additional_params = get_dataloaders(model_name, b_size, data_name)

    optimizer_hparams = {"lr": l_rate, "weight_decay": w_decay, "warmup": warmup, "max_iters": max_iters}

    model_params = {
        'model': model_name,
        'cf_hid_dim': cf_hidden_dim,
        **additional_params
    }

    trainer = initialize_trainer(epochs, patience, model_name, l_rate, w_decay, warmup, seed, data_name)

    # optionally resume from a checkpoint
    if resume is not None:
        print(f'=> intending to resume from checkpoint')
        if os.path.isfile(resume):
            print(f"=> loading checkpoint '{resume}'")
            model = DocumentClassifier.load_from_checkpoint(resume)
            print(f"=> loaded checkpoint '{resume}'\n")
        else:
            raise ValueError(f"No checkpoint found at '{resume}'!")
    else:
        model = DocumentClassifier(model_params, optimizer_hparams)

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


def initialize_trainer(epochs, patience, model_name, l_rate, weight_decay, warmup, seed, dataset):
    model_checkpoint = cb.ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_accuracy")

    os.makedirs(LOG_PATH, exist_ok=True)

    version_str = f'dname={dataset}_seed={seed}_lr={l_rate}_wdec={weight_decay}_wsteps={warmup}'
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
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=1)
    parser.add_argument('--lr', dest='l_rate', type=float, default=0.01)
    parser.add_argument("--w-decay", dest='w_decay', type=float, default=2e-3,
                        help="Weight decay for L2 regularization of optimizer AdamW")
    parser.add_argument("--warmup", dest='warmup', type=int, default=500,
                        help="Number of steps for which we do learning rate warmup.")
    parser.add_argument("--max-iters", dest='max_iters', type=int, default=2000,
                        help="Max iterations for learning rate warmup.")

    # CONFIGURATION

    parser.add_argument('--dataset', dest='dataset', default='R8', choices=SUPPORTED_DATASETS,
                        help='Select the dataset you want to use.')
    parser.add_argument('--model', dest='model', default='roberta_gnn', choices=SUPPORTED_MODELS,
                        help='Select the model you want to use.')
    parser.add_argument('--seed', dest='seed', type=int, default=1234)
    parser.add_argument('--cf-hidden-dim', dest='cf_hidden_dim', type=int, default=512)
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: None)')

    params = vars(parser.parse_args())

    train(
        model_name=params['model'],
        seed=params['seed'],
        epochs=params['epochs'],
        patience=params['patience'],
        b_size=params["batch_size"],
        l_rate=params["l_rate"],
        w_decay=params["w_decay"],
        warmup=params["warmup"],
        max_iters=params["max_iters"],
        cf_hidden_dim=params["cf_hidden_dim"],
        data_name=params["dataset"],
        resume=params["resume"]
    )
