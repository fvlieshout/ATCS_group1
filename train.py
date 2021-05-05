import argparse
import os
import time

import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast
from transformers.data.data_collator import default_data_collator

from data_prep.reuters_text import R8, R52
from models.model import ClassifierModule

# disable parallelism for hugging face to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# more specific cuda errors
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

LOG_PATH = "./logs/"

SUPPORTED_MODELS = ['roberta']
SUPPORTED_DATASETS = ['R8', 'R52']


def train(model_name, seed, epochs, b_size, l_rate, l_decay, minimum_lr, cf_hidden_dim, dataset_name='R8'):
    os.makedirs(LOG_PATH, exist_ok=True)

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

    model_params = {'model': model_name, "num_classes": 8 if dataset == R8 else 52, "cf_hid_dim": cf_hidden_dim}
    optimizer_hparams = {"lr": l_rate, "weight_decay": l_decay}

    model = ClassifierModule(model_params, optimizer_hparams)

    trainer = initialize_trainer(epochs, minimum_lr, model_name)

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
    return DataLoader(dataset, batch_size=b_size, num_workers=24, shuffle=shuffle, collate_fn=default_data_collator)


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
    print(f'Test Results:\n test accuracy: {test_accuracy}\n validation accuracy: {val_accuracy}\n')

    return test_accuracy, val_accuracy


def initialize_trainer(epochs, minimum_lr, model):
    model_checkpoint = cb.ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_accuracy")

    log_dir = os.path.join(LOG_PATH, model)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir + "/lightning_logs", exist_ok=True)

    trainer = pl.Trainer(default_root_dir=log_dir,
                         checkpoint_callback=model_checkpoint,
                         gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs=epochs,
                         callbacks=[LearningRateMonitor("epoch"),
                                    LearningRateStopping(min_value=minimum_lr)],
                         progress_bar_refresh_rate=1)

    # Optional logging argument that we don't need
    trainer.logger._default_hp_metric = None

    return trainer


def get_dataset(dataset_name):
    if dataset_name == "R8":
        return R8
    elif dataset_name == "R52":
        return R52
    else:
        raise ValueError("Dataset '%s' is not supported." % dataset_name)


class LearningRateStopping(pl.Callback):

    def __init__(self, min_value):
        super().__init__()
        self.min_value = min_value

    def on_validation_end(self, trainer, pl_module):
        current_lr = trainer.optimizers[0].param_groups[0]['lr']
        if current_lr is not None and current_lr <= self.min_value:
            print('Stopping training current LR ' + str(current_lr) + ' min LR ' + str(self.min_value))
            trainer.should_stop = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # TRAINING PARAMETERS

    parser.add_argument('--epochs', dest='epochs', type=int, default=3)
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=2)
    parser.add_argument('--lr', dest='l_rate', type=float, default=1e-4)
    parser.add_argument("--min-lr", dest='minimum_lr', type=float, default=1e-5, help="Minimum Learning Rate")
    parser.add_argument("--lr-decay", dest='lr_decay', type=float, default=1e-3, help="Learning rate (weight) decay")

    # CONFIGURATION

    parser.add_argument('--dataset', dest='dataset', default='R8', choices=SUPPORTED_DATASETS,
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
        b_size=params["batch_size"],
        l_rate=params["l_rate"],
        l_decay=params["lr_decay"],
        minimum_lr=params["minimum_lr"],
        cf_hidden_dim=params["cf_hidden_dim"],
        dataset_name=params["dataset"]
    )
