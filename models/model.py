import numpy as np
import pytorch_lightning as pl
from models.glove_graph_encoder import GloveGraphEncoder
from models.roberta_encoder import RobertaEncoder
from models.roberta_graph_encoder import RobertaGraphEncoder
from numpy.lib.arraysetops import isin
from torch import nn
from torch import optim
from torch.optim import AdamW


class DocumentClassifier(pl.LightningModule):

    # noinspection PyUnusedLocal
    def __init__(self, model_hparams, optimizer_hparams):
        """
        Inputs:
            model_hparams - Hyperparameters for the whole model, as dictionary. Also contains Roberta Configuration.
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate,
            weight decay, etc.
        """
        super().__init__()

        # Variable to distinguish between validation and test mask in test_step
        self.test_val_mode = 'test'

        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        self.loss_module = nn.CrossEntropyLoss()

        roberta_output_dim = 768
        model_name = model_hparams['model']
        if 'checkpoint' in model_hparams and model_hparams['checkpoint'] is not None:
            self.model = load_pretrained_encoder(model_hparams['checkpoint'])
        elif model_name == 'roberta':
            self.model = RobertaEncoder()
        elif model_name == 'glove_gnn':
            self.model = GloveGraphEncoder(model_hparams['doc_dim'], model_hparams['word_dim'], roberta_output_dim)
        elif model_name == 'roberta_gnn':
            self.model = RobertaGraphEncoder(roberta_output_dim, roberta_output_dim)
        else:
            raise ValueError("Model type '%s' is not supported." % model_name)

        cf_hidden_dim = model_hparams['cf_hid_dim']

        self.classifier = nn.Sequential(
            # nn.Dropout(model_hparams['dropout']),     # TODO: maybe add later
            nn.Linear(roberta_output_dim, cf_hidden_dim),
            nn.ReLU(),
            nn.Linear(cf_hidden_dim, model_hparams['num_classes'])
        )

        self.lr_scheduler = None

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
        out, labels = self.model(batch, mode='train')
        predictions = self.classifier(out)
        loss = self.loss_module(predictions, labels)

        self.log('train_accuracy', self.accuracy(predictions, labels).item(), on_step=False, on_epoch=True)
        self.log('train_loss', loss)

        # logging in optimizer step does not work, therefore here
        self.log('lr_rate', self.lr_scheduler.get_lr()[0])
        return loss

    def validation_step(self, batch, batch_idx):
        # By default logs it per epoch (weighted average over batches)
        out, labels = self.model(batch, mode='val')
        predictions = self.classifier(out)
        self.log('val_accuracy', self.accuracy(predictions, labels))

    def test_step(self, batch, batch_idx):
        # By default logs it per epoch (weighted average over batches)
        out, labels = self.model(batch, mode=self.test_val_mode)
        predictions = self.classifier(out)
        self.log('test_accuracy', self.accuracy(predictions, labels))

    def backward(self, loss, optimizer, optimizer_idx):
        # override backward pass so we can set retain_graph to True; should be true for the roberta graph encoder
        loss.backward(retain_graph=isinstance(self.model, RobertaGraphEncoder))

    @staticmethod
    def accuracy(predictions, labels):
        # noinspection PyUnresolvedReferences
        return (labels == predictions.argmax(dim=-1)).float().mean()


# noinspection PyProtectedMember
class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

def load_pretrained_encoder(checkpoint_path):
    module = DocumentClassifier.load_from_checkpoint(checkpoint_path)
    module.freeze()
    return module.model