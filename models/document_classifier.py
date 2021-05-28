import numpy as np
import pytorch_lightning as pl
import torch
from models.glove_graph_encoder import GloveGraphEncoder
from models.roberta_encoder import RobertaEncoder
from models.roberta_graph_encoder import RobertaGraphEncoder
from numpy.lib.arraysetops import isin
from torch import nn
from torch import optim
from torch.optim import AdamW


class DocumentClassifier(pl.LightningModule):
    """
    PyTorch Lightning module containing all model setup: Picking the correct encoder, initializing the classifier,
    and overwriting standard functions for training and optimization.
    """

    # noinspection PyUnusedLocal
    def __init__(self, model_hparams, optimizer_hparams, checkpoint=None, transfer=False, h_search=False):
        """
        Args:
            model_hparams - Hyperparameters for the whole model, as dictionary. Also contains Roberta Configuration.
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate,
            weight decay, etc.
        """
        super().__init__()

        if transfer and checkpoint is None:
            raise ValueError("Missing checkpoint path.")

        # Variable to distinguish between validation and test mask in test_step
        self.test_val_mode = 'test'

        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        self.loss_module = nn.CrossEntropyLoss()

        roberta_output_dim = 768
        model_name = model_hparams['model']

        if model_name == 'roberta':
            self.model = RobertaEncoder(h_search)
        elif model_name == 'glove_gnn':
            self.model = GloveGraphEncoder(
                model_hparams['doc_dim'], model_hparams['word_dim'], roberta_output_dim,
                model_hparams['gnn_layer_name'])
        elif model_name in ['roberta_pretrained_gnn', 'roberta_finetuned_gnn']:
            self.model = RobertaGraphEncoder(roberta_output_dim, roberta_output_dim, model_hparams['gnn_layer_name'])
        else:
            raise ValueError("Model type '%s' is not supported." % model_name)

        if transfer:
            encoder = load_pretrained_encoder(checkpoint)
            self.model.load_state_dict(encoder)

        cf_hidden_dim = model_hparams['cf_hid_dim']

        self.classifier = nn.Sequential(
            nn.Linear(roberta_output_dim, cf_hidden_dim),
            nn.ReLU(),
            nn.Linear(cf_hidden_dim, model_hparams['num_classes'])
        )

        self.lr_scheduler = None

    def configure_optimizers(self):
        """
        Configures the AdamW optimizer and enables training with different learning rates for encoder and classifier.
        Also initializes the learning rate scheduler.
        """

        lr_enc = self.hparams.optimizer_hparams['lr_enc']
        lr_cl = self.hparams.optimizer_hparams['lr_cl']
        if lr_cl < 0:  # classifier learning rate not specified
            lr_cl = lr_enc

        weight_decay_enc = self.hparams.optimizer_hparams["weight_decay_enc"]
        weight_decay_cl = self.hparams.optimizer_hparams["weight_decay_cl"]
        if weight_decay_cl < 0:  # classifier weight decay not specified
            weight_decay_cl = weight_decay_enc

        params = list(self.named_parameters())

        def is_encoder(n):
            return n.startswith('model')

        grouped_parameters = [
            {
                'params': [p for n, p in params if is_encoder(n)],
                'lr': lr_enc,
                'weight_decay': weight_decay_enc
            },
            {
                'params': [p for n, p in params if not is_encoder(n)],
                'lr': lr_cl,
                'weight_decay': weight_decay_cl
            }
        ]

        optimizer = AdamW(grouped_parameters)

        self.lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=self.hparams.optimizer_hparams['warmup'],
                                                  max_iters=self.hparams.optimizer_hparams['max_iters'])

        return [optimizer], []

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration

    def training_step(self, batch, _):
        out, labels = self.model(batch, mode='train')
        predictions = self.classifier(out)
        loss = self.loss_module(predictions, labels)

        self.log('train_accuracy', self.accuracy(predictions, labels).item(), on_step=False, on_epoch=True)
        self.log('train_loss', loss)

        # logging in optimizer step does not work, therefore here
        self.log('lr_rate', self.lr_scheduler.get_lr()[0])
        return loss

    def validation_step(self, batch, _):
        # By default logs it per epoch (weighted average over batches)
        out, labels = self.model(batch, mode='val')
        predictions = self.classifier(out)
        self.log('val_accuracy', self.accuracy(predictions, labels))

    def test_step(self, batch, _):
        # By default logs it per epoch (weighted average over batches)
        out, labels = self.model(batch, mode=self.test_val_mode)
        predictions = self.classifier(out)
        self.log('test_accuracy', self.accuracy(predictions, labels))

    def backward(self, loss, *_):
        """
        Overrides backward pass in order to set retain_graph to True.
        Roberta graph encoder requires the graph to be retained.
        """
        loss.backward(retain_graph=isinstance(self.model, RobertaGraphEncoder))

    @staticmethod
    def accuracy(predictions, labels):
        # noinspection PyUnresolvedReferences
        return (labels == predictions.argmax(dim=-1)).float().mean()


# noinspection PyProtectedMember
class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler, combining warm-up with a cosine-shaped learning rate decay.
    """

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor()
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self):
        current_step = self.last_epoch
        lr_factor = 0.5 * (1 + np.cos(np.pi * current_step / self.max_num_iters))
        if current_step < self.warmup:
            lr_factor *= current_step * 1.0 / self.warmup
        return lr_factor


def load_pretrained_encoder(checkpoint_path):
    """
    Load a pretrained encoder state dict and remove 'model.' from the keys in the state dict, so that solely
    the encoder can be loaded.

    Args:
        checkpoint_path (str) - Path to a checkpoint for the DocumentClassifier.
    Returns:
        encoder_state_dict (dict) - Containing all keys for weights of encoder.
    """
    checkpoint = torch.load(checkpoint_path)
    encoder_state_dict = {}
    for layer, param in checkpoint["state_dict"].items():
        if layer.startswith("model"):
            new_layer = layer[layer.index(".") + 1:]
            encoder_state_dict[new_layer] = param

    return encoder_state_dict
