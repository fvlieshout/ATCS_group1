import argparse
import os
import time

import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
import torch
from torch.utils.data import DataLoader
from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers.data.data_collator import default_data_collator

from datasets.reuters_text import R8
from model import RobertaModule

# disable parallelism for hugging face to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

LOG_PATH = "./logs/"

SUPPORTED_MODELS = ['baseline']


def train(model, seed, epochs, b_size, l_rate, dataset=R8):
    os.makedirs(LOG_PATH, exist_ok=True)

    pl.seed_everything(seed)

    if model == 'baseline':

        # Prepare the data

        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

        train_dataset, test_dataset, val_dataset = dataset.splits(tokenizer, val_size=0.1)

        train_dataloader = data_loader(b_size, train_dataset, shuffle=True)
        test_dataloader = data_loader(b_size, test_dataset)
        val_dataloader = data_loader(b_size, val_dataset)

        # Prepare the model

        configuration = RobertaConfig(
            vocab_size=tokenizer.vocab_size,
            max_position_embeddings=514,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1,
        )

        model_params = {"num_classes": 8 if dataset == R8 else 52,
                        "cf_hid_dim": 256,
                        'roberta_config': configuration
                        }

        optimizer_hparams = {"lr": l_rate}

        model = RobertaModule(model_params, optimizer_hparams)

    else:
        raise ValueError("Model type '%s' is not supported." % model)

    trainer = initialize_trainer(epochs)

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


def initialize_trainer(epochs, log_dir=None):
    model_checkpoint = cb.ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_accuracy")

    # lr_monitor = LearningRateMonitor("epoch")

    trainer = pl.Trainer(default_root_dir=log_dir,
                         checkpoint_callback=model_checkpoint,
                         gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs=epochs,
                         callbacks=[],
                         progress_bar_refresh_rate=1)

    # Optional logging argument that we don't need
    trainer.logger._default_hp_metric = None

    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # TRAINING PARAMETERS

    parser.add_argument('--epochs', dest='epochs', type=int, default=3)
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=2)
    parser.add_argument('--lr', dest='l_rate', type=float, default=0.1)

    # CONFIGURATION

    parser.add_argument('--model', dest='model', default='baseline', choices=SUPPORTED_MODELS,
                        help='Select the model you want to use.')
    parser.add_argument('--seed', dest='seed', type=int, default=1234)

    params = vars(parser.parse_args())

    train(params['model'],
          params['seed'],
          params['epochs'],
          params["batch_size"],
          params["l_rate"])
