import pytorch_lightning as pl
from torch import nn
from torch.optim import Adam
from transformers import RobertaModel as Roberta


class RobertaModule(pl.LightningModule):

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

        self.model = RobertaModel(model_hparams)

    def forward(self, inputs):
        return self.model(inputs)

    def configure_optimizers(self):
        return Adam(self.parameters(), **self.hparams.optimizer_hparams)

    def training_step(self, batch, batch_idx):
        """
        Inputs:
            batch         - Input batch, output of the training loader.
            batch_idx     - Index of the batch in the dataset (not needed here).
        """

        # "batch" is the output of the training data loader
        predictions, labels = self.predict(batch)
        loss = self.loss_module(predictions, labels)

        self.log('train_accuracy', self.accuracy(predictions, labels).item(), on_step=False, on_epoch=True)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        # By default logs it per epoch (weighted average over batches)
        self.log('val_accuracy', self.accuracy(*self.predict(batch)))

    def test_step(self, batch, batch_idx):
        # By default logs it per epoch (weighted average over batches)
        self.log('test_accuracy', self.accuracy(*self.predict(batch)))

    @staticmethod
    def accuracy(predictions, labels):
        # noinspection PyUnresolvedReferences
        return (labels == predictions.argmax(dim=-1)).float().mean()

    def predict(self, batch):
        return self.model(batch['input_ids']), batch['labels']

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RobertaModel(nn.Module):

    def __init__(self, model_hparams):
        super().__init__()

        roberta_config = model_hparams['roberta_config']
        self.roberta = Roberta(roberta_config)

        # Freeze roberta parameters TODO: freeze only half of this and fine tune the other half
        for param in self.roberta.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(roberta_config.hidden_size, model_hparams['cf_hid_dim']),
            nn.Linear(model_hparams['cf_hid_dim'], model_hparams['num_classes'])
        )

    def forward(self, inputs):
        # returns a tuple of torch.FloatTensor comprising various elements depending on the (RobertaConfig) and inputs.
        roberta_output = self.roberta(inputs)

        # b_size x seq_len x hid_size
        # last_hidden_state = roberta_output[0]

        # b_size x hid_size
        cls_token_state = roberta_output[1]

        # resulting vector: fed into classifier
        predictions = self.classifier(cls_token_state)
        return predictions
