import argparse
import os
import time

import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
import torch
import torch_geometric.data as geom_data
from data_prep.agnews_text import AGNewsText
from data_prep.reuters_graph import R8Graph, R52Graph
from data_prep.reuters_text import R8Text, R52Text
from models.model import ClassifierModule, TransformerClassifier
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast


# disable parallelism for hugging face to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# more specific cuda errors
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

LOG_PATH = "./logs/"

SUPPORTED_MODELS = ['roberta', 'pure-gnn']
SUPPORTED_DATASETS = ['R8Text', 'R52Text', 'R8Graph', 'R52Graph', 'AGNewsText', 'AGNewsGraph']

from torch import nn
from torch import optim
from torch.optim import AdamW
from models.model import CosineWarmupScheduler
class TransferClassifierModule(pl.LightningModule):

    # noinspection PyUnusedLocal
    def __init__(self, model_hparams, optimizer_hparams):
        """
        Inputs:
            model_hparams - Hyperparameters for the whole model, as dictionary. Also contains Roberta Configuration.
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate,
            weight decay, etc.
        """
        super().__init__()

        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()

        self.t = optimizer_hparams

        self.loss_module = nn.CrossEntropyLoss()

        ClassifierModule

        self.model = ClassifierModule.load_from_checkpoint(model_hparams['checkpoint'])
        self.model.freeze()

        n_out_features = self.model.model.classifier[0].in_features

        self.classifier = nn.Linear(in_features=n_out_features, out_features=model_hparams["num_classes"])

        print(self.classifier)

        self.lr_scheduler = None

    def forward(self, batch, mode):
        inputs, attention_mask = batch['input_ids'], batch['attention_mask']
        out = self.model.model.encoder(inputs, attention_mask)
        out = self.classifier(out)
        return out, batch["labels"]

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.optimizer_hparams['lr'],
                          weight_decay=self.hparams.optimizer_hparams['weight_decay'])

        self.lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=self.hparams.optimizer_hparams['warmup'],
                                                  max_iters=self.hparams.optimizer_hparams['max_iters'])

        return [optimizer], []

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration

    def training_step(self, batch, batch_idx):
        """
        Inputs:
            batch         - Input batch, output of the training loader.
            batch_idx     - Index of the batch in the dataset (not needed here).
        """
        # "batch" is the output of the training data loader
        predictions, labels = self.forward(batch, mode='train')
        loss = self.loss_module(predictions, labels)

        # print('loss ' + str(loss.item()))

        self.log('train_accuracy', self.accuracy(predictions, labels).item(), on_step=False, on_epoch=True)
        self.log('train_loss', loss)

        # logging in optimizer step does not work, therefore here
        self.log('lr_rate', self.lr_scheduler.get_lr()[0])

        return loss

    def validation_step(self, batch, batch_idx):
        # By default logs it per epoch (weighted average over batches)
        self.log('val_accuracy', self.accuracy(*self.forward(batch, mode='val')))

    def test_step(self, batch, batch_idx):
        # By default logs it per epoch (weighted average over batches)
        self.log('test_accuracy', self.accuracy(*self.forward(batch, mode='test')))

    @staticmethod
    def accuracy(predictions, labels):
        # noinspection PyUnresolvedReferences
        return (labels == predictions.argmax(dim=-1)).float().mean()

def train(model_name, seed, epochs, patience, b_size, l_rate, w_decay, warmup, max_iters, cf_hidden_dim, dataset_name):
    os.makedirs(LOG_PATH, exist_ok=True)

    if model_name not in SUPPORTED_MODELS:
        raise ValueError("Model type '%s' is not supported." % model_name)

    print(f'Configuration:\n model_name: {model_name}\n max epochs: {epochs}\n patience: {patience}'
          f'\n seed: {seed}\n batch_size: {b_size}\n l_rate: {l_rate}\n warmup: {warmup}\n '
          f'weight_decay: {w_decay}\n cf_hidden_dim: {cf_hidden_dim}\n dataset_name: {dataset_name}\n')

    pl.seed_everything(seed)

    # the data preprocessing

    train_loader, test_loader, val_loader, additional_params = get_dataloaders(model_name, b_size, dataset_name)

    optimizer_hparams = {"lr": l_rate, "weight_decay": w_decay, "warmup": warmup, "max_iters": max_iters}

    model_params = {
        'model': model_name,
        'cf_hid_dim': cf_hidden_dim,
        **additional_params
    }

    model = ClassifierModule(model_params, optimizer_hparams)
    trainer = initialize_trainer(epochs, patience, model_name, l_rate, w_decay, warmup)

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


def get_dataloaders(model, b_size, dataset_name):
    dataset = get_dataset(dataset_name)
    additional_params = {}

    if model == 'roberta':
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        train_dataset, test_dataset, val_dataset = dataset.splits(tokenizer, val_size=0.1)

        additional_params['num_classes'] = train_dataset.num_classes

        train_dataloader = text_dataloader(train_dataset, b_size, shuffle=True)
        test_dataloader = text_dataloader(test_dataset, b_size)
        val_dataloader = text_dataloader(val_dataset, b_size)

    elif model == 'pure-gnn':
        train_dataloader = geom_data.DataLoader(dataset, batch_size=1)
        val_dataloader = geom_data.DataLoader(dataset, batch_size=1)
        test_dataloader = geom_data.DataLoader(dataset, batch_size=1)

        additional_params['num_nodes'] = len(dataset.iton)
    else:
        raise ValueError("Model type '%s' is not supported." % model)

    return train_dataloader, test_dataloader, val_dataloader, additional_params


def text_dataloader(dataset, b_size, shuffle=False):
    return DataLoader(dataset, batch_size=b_size, num_workers=24, shuffle=shuffle, collate_fn=dataset.get_collate_fn())


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


def initialize_trainer(epochs, patience, model_name, l_rate, weight_decay, warmup):
    model_checkpoint = cb.ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_accuracy")

    os.makedirs(LOG_PATH, exist_ok=True)

    version_str = f'patience={patience}_lr={l_rate}_wdec={weight_decay}_wsteps={warmup}'
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
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if dataset_name == "R8Text":
        return R8Text
    elif dataset_name == "R52Text":
        return R52Text
    elif dataset_name == "AGNewsText":
        return AGNewsText
    elif dataset_name == 'R8Graph':
        return R8Graph(device)
    elif dataset_name == 'R52Graph':
        return R52Graph(device)
    else:
        raise ValueError("Dataset '%s' is not supported." % dataset_name)


def eval(model_name, seed, epochs, patience, b_size, l_rate, w_decay, warmup, max_iters, cf_hidden_dim, dataset_name):
    os.makedirs(LOG_PATH, exist_ok=True)

    if model_name not in SUPPORTED_MODELS:
        raise ValueError("Model type '%s' is not supported." % model_name)

    print(f'Configuration:\n model_name: {model_name}\n max epochs: {epochs}\n patience: {patience}'
          f'\n seed: {seed}\n batch_size: {b_size}\n l_rate: {l_rate}\n warmup: {warmup}\n '
          f'weight_decay: {w_decay}\n cf_hidden_dim: {cf_hidden_dim}\n dataset_name: {dataset_name}\n')

    pl.seed_everything(seed)

    # the data preprocessing

    train_loader, test_loader, val_loader, additional_params = get_dataloaders(model_name, b_size, dataset_name)

    optimizer_hparams = {"lr": l_rate, "weight_decay": w_decay, "warmup": warmup, "max_iters": max_iters}

    model_params = {
        'model': model_name,
        'cf_hid_dim': cf_hidden_dim,
        **additional_params
    }

    model = ClassifierModule(model_params, optimizer_hparams)
    trainer = initialize_trainer(epochs, patience, model_name, l_rate, w_decay, warmup)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # TRAINING PARAMETERS

    parser.add_argument('--epochs', dest='epochs', type=int, default=50)
    parser.add_argument('--patience', dest='patience', type=int, default=10)
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=64)
    parser.add_argument('--lr', dest='l_rate', type=float, default=1e-4)
    parser.add_argument("--min-lr", dest='minimum_lr', type=float, default=1e-5, help="Minimum Learning Rate")
    parser.add_argument("--w-decay", dest='w_decay', type=float, default=1e-3,
                        help="Weight decay for L2 regularization of optimizer AdamW")
    parser.add_argument("--warmup", dest='warmup', type=int, default=100,
                        help="Number of steps for which we do learning rate warmup.")
    parser.add_argument("--max-iters", dest='max_iters', type=int, default=2000,
                        help="Max iterations for learning rate warmup.")

    # CONFIGURATION

    parser.add_argument('--checkpoint', dest='checkpoint', help='Path to the checkpoint file.')
    parser.add_argument('--dataset', dest='dataset', default='R8Graph', choices=SUPPORTED_DATASETS,
                        help='Select the dataset you want to use.')
    parser.add_argument('--model', dest='model', default='pure-gnn', choices=SUPPORTED_MODELS,
                        help='Select the model you want to use.')
    parser.add_argument('--seed', dest='seed', type=int, default=1234)
    parser.add_argument('--cf-hidden-dim', dest='cf_hidden_dim', type=int, default=512)

    params = vars(parser.parse_args())

    # train(
    #     model_name=params['model'],
    #     seed=params['seed'],
    #     epochs=params['epochs'],
    #     patience=params['patience'],
    #     b_size=params["batch_size"],
    #     l_rate=params["l_rate"],
    #     w_decay=params["w_decay"],
    #     warmup=params["warmup"],
    #     max_iters=params["max_iters"],
    #     cf_hidden_dim=params["cf_hidden_dim"],
    #     dataset_name=params["dataset"]
    # )
    print(params)

    train_loader, test_loader, val_loader, additional_params = get_dataloaders(params["model"], params["batch_size"], params["dataset"])

    optimizer_hparams = {"lr": params["l_rate"], "weight_decay": params["w_decay"], "warmup": params["warmup"], "max_iters": params["max_iters"]}

    model_params = {
        'model': params["model"],
        'cf_hid_dim': params["cf_hidden_dim"],
        'checkpoint': params["checkpoint"],
        **additional_params
    }
    print(model_params)

    transfer_model = TransferClassifierModule(model_params, optimizer_hparams)

    trainer = initialize_trainer(5, 2, "roberta", params['l_rate'], params['w_decay'], params['warmup'])

    # Training
    print('Fitting model ..........\n')
    trainer.fit(transfer_model, train_loader, val_loader)

    # Testing
    # Load best checkpoint after training
    test_acc, val_acc = evaluate(trainer, transfer_model, test_loader, val_loader)


    


