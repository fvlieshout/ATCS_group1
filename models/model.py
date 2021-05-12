import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
from torch.optim import AdamW
from transformers import RobertaModel
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


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

    def forward(self, batch):
        return self.model(batch['input_ids'], batch['attention_mask']), batch['labels']

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), **self.hparams.optimizer_hparams)

        # Disabling it for now as it prevented the model somehow to actually learn
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1,
        #                                       gamma=self.hparams.optimizer_hparams['weight_decay'])

        return [optimizer], [] #[scheduler]

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

class GraphNet(torch.nn.Module):
    def __init__(self):
        super(GraphNet, self).__init__()
        self.linlay = nn.Linear(300, 768)
        self.conv1 = GCNConv(768, 200)
        self.conv2 = GCNConv(200, 8)

    def forward(self, data, device):
        features, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        doc_feats = features[:data.num_docs]
        word_feats = features[data.num_docs:,:300]
        word_feats = self.linlay(word_feats)
        x = torch.cat((doc_feats, word_feats))
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

class GraphModel(pl.LightningModule):
    def __init__(self, num_nodes, optimizer_hparams):
        super().__init__()
        self.devc = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.model = GraphNet().to(self.devc)
        self.test_val_mode = 'test'
        self.save_hyperparameters()
    
    def forward(self, data, mode):
        out = self.model(data, self.devc)
        if mode=='train':
            mask = data.train_mask
        elif mode=='val':
            mask = data.val_mask
        elif mode=='test':
            mask = data.test_mask
        loss = F.cross_entropy(out[mask], data.y[mask])
        class_predictions = torch.argmax(out, dim=1)
        if mode=='val':
            print('preds:', class_predictions[:20])
            print('real:', data.y[:20])
        correct = (class_predictions[mask] == data.y[mask]).sum().item()
        accuracy = correct / mask.sum()
        return loss, accuracy
    
    def configure_optimizers(self):
        self.optimizer = AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        return self.optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode='train')
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_accuracy", acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode='val')
        # self.log("val_loss", loss)
        self.log("val_accuracy", acc)

    def test_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode=self.test_val_mode)
        # self.log("test_loss", loss)
        self.log("test_accuracy", acc)
