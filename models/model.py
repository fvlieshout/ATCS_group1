import math

import pytorch_lightning as pl
import pytorch_warmup as warmup
from torch import nn
from torch import optim
from torch.optim import AdamW
from transformers import RobertaModel


class ClassifierModule(pl.LightningModule):

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

        self.loss_module = nn.CrossEntropyLoss()

        self.model = DocumentClassifier(model_hparams)

        # self.warmup_scheduler = None

        self.warmup_phase = 1000

    def forward(self, batch):
        return self.model(batch['input_ids'], batch['attention_mask']), batch['labels']

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.optimizer_hparams['lr'],
                          weight_decay=self.hparams.optimizer_hparams['weight_decay'])

        # Disabling it for now as it prevented the model somehow to actually learn
        step_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1,
                                                   gamma=self.hparams.optimizer_hparams['lr_decay'])
        # warmup_scheduler = LearningRateWarmup(optimizer)

        # self.warmup_scheduler = warmup.RAdamWarmup(optimizer)

        return [optimizer], [step_scheduler]

    def training_step(self, batch, batch_idx):
        """
        Inputs:
            batch         - Input batch, output of the training loader.
            batch_idx     - Index of the batch in the dataset (not needed here).
        """
        # "batch" is the output of the training data loader
        predictions, labels = self.forward(batch)
        loss = self.loss_module(predictions, labels)

        self.log('train_accuracy', self.accuracy(predictions, labels).item(), on_step=False, on_epoch=True)
        self.log('train_loss', loss)

        return loss

    def training_step_end(self, training_step_outputs):
        # TODO: call learning rate warmup

        lr = self.optimizers().param_groups[0]['lr']
        # print('global Step ' + str(self.global_step))
        # print('LR before ' + str(lr))

        # self.warmup_scheduler.dampen()
        omega = (1 - math.exp(-self.global_step / self.warmup_phase))
        # print('warmup factor ' + str(omega))

        if omega != 0:
            self.optimizers().param_groups[0]['lr'] = omega * self.hparams.optimizer_hparams['lr']

        # print('LR after ' + str(self.optimizers().param_groups[0]['lr']))

    def validation_step(self, batch, batch_idx):
        # By default logs it per epoch (weighted average over batches)
        self.log('val_accuracy', self.accuracy(*self.forward(batch)))

    def test_step(self, batch, batch_idx):
        # By default logs it per epoch (weighted average over batches)
        self.log('test_accuracy', self.accuracy(*self.forward(batch)))

    @staticmethod
    def accuracy(predictions, labels):
        # noinspection PyUnresolvedReferences
        return (labels == predictions.argmax(dim=-1)).float().mean()


class DocumentClassifier(nn.Module):

    def __init__(self, model_hparams):
        super().__init__()

        model = model_hparams['model']
        if model == 'roberta':
            self.encoder = TransformerModel()
        else:
            raise ValueError("Model type '%s' is not supported." % model)

        cf_hidden_dim = model_hparams['cf_hid_dim']

        self.classifier = nn.Sequential(
            # nn.Dropout(model_hparams['dropout']),     # TODO: maybe add later
            nn.Linear(self.encoder.hidden_size, cf_hidden_dim),
            nn.ReLU(),
            nn.Linear(cf_hidden_dim, model_hparams['num_classes'])
        )

    def forward(self, inputs, attention_mask=None):
        out = self.encoder(inputs, attention_mask)
        out = self.classifier(out)
        return out


# class LearningRateWarmup(optim.lr_scheduler._LRScheduler):
#
#     def __init__(self, optimizer, warmup_epochs=10, warmup_start_lr=0.0, last_epoch=-1, verbose=False):
#         self.warmup_epochs = warmup_epochs
#         self.warmup_start_lr = warmup_start_lr
#         self.count = 0
#         super(LearningRateWarmup, self).__init__(optimizer, last_epoch, verbose)
#
#     def get_lr(self):
#         if not self._get_lr_called_within_step:
#             warnings.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.",
#                           UserWarning)
#
#         if self.last_epoch > self.warmup_epochs:
#             return [group['lr'] for group in self.optimizer.param_groups]
#
#         print('LR before ' + str(self.optimizer.param_groups[0]['lr']))
#
#         self.count += 1
#         omega = 1.0 - math.exp(-self.count / self.warmup_epochs)
#
#         print('LR before ' + str(self.optimizer.param_groups[0]['lr'] * omega))
#
#         return [group['lr'] * omega for group in self.optimizer.param_groups]
#
#     # def _get_closed_form_lr(self):
#     #     return [base_lr * self.gamma ** (self.last_epoch // self.step_size)
#     #             for base_lr in self.base_lrs]


class TransformerModel(nn.Module):

    def __init__(self):
        super().__init__()

        # transformer_config = model_hparams['transformer_config']
        self.model = RobertaModel.from_pretrained('roberta-base')

        # this is fixed for all base models
        self.hidden_size = 768

        # this model is in eval per default, we want to fine-tune it but only the top layers
        self.model.train()

        # only freezing the encoder parameters / weights of the head layers
        for param in self.model.base_model.parameters():
            param.requires_grad = False

    def forward(self, inputs, attention_mask=None):
        # returns a tuple of torch.FloatTensor comprising various elements depending on the (RobertaConfig) and inputs.
        hidden_states = self.model(inputs, attention_mask)

        # b_size x hid_size
        cls_token_state = hidden_states[1]

        return cls_token_state
