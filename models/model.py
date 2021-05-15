import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.optim import AdamW
from torch_geometric.nn import GCNConv
from transformers import RobertaModel


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
        pure_gnn_output_dim = model_hparams['gnn_output_dim']
        cf_hidden_dim = model_hparams['cf_hid_dim']

        model_name = model_hparams['model']
        if model_name == 'roberta':
            self.model = RobertaEncoder()
        elif model_name == 'pure_gnn':
            self.model = PureGraphEncoder(pure_gnn_output_dim, roberta_output_dim)
        elif model_name == 'roberta_gnn':
            self.model = RobertaGraphEncoder(roberta_output_dim, roberta_output_dim)
        else:
            raise ValueError("Model type '%s' is not supported." % model_name)
        
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
        #override of the backward pass so we can set retain_graph to True
        loss.backward(retain_graph=True)

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


class RobertaEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # transformer_config = model_hparams['transformer_config']
        self.model = RobertaModel.from_pretrained('roberta-base')

        # this model is in eval per default, we want to fine-tune it but only the top layers
        self.model.train()

        # only freezing the encoder parameters / weights of the head layers
        for param in self.model.base_model.parameters():
            param.requires_grad = False

    def forward(self, batch):
        inputs, attention_mask = batch['input_ids'], batch['attention_mask']
        # returns a tuple of torch.FloatTensor comprising various elements depending on the (RobertaConfig) and inputs.
        hidden_states = self.model(inputs, attention_mask)

        # b_size x hid_size
        out = hidden_states[1]
        return out, batch['labels']

class PureGraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PureGraphEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, data, mode):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        edge_index, edge_weight = data.edge_index, data.edge_attr
        doc_feats = data.doc_features
        word_feats = data.word_features
        word_feats = self.linlay(word_feats)
        x = torch.cat((doc_feats, word_feats))
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        if mode == 'train':
            mask = data.train_mask
        elif mode == 'val':
            mask = data.val_mask
        elif mode == 'test':
            mask = data.test_mask
        else:
            raise ValueError("Mode '%s' is not supported in forward of graph classifier." % mode)

        x = x[mask]
        return x, data.y[mask]

class RobertaGraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RobertaGraphEncoder, self).__init__()
        self.linlay = nn.Linear(300, 768)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, data, mode):
        #the batch is a Data object here
        edge_index, edge_weight = data.edge_index, data.edge_attr
        doc_feats = data.doc_features
        word_feats = data.word_features
        word_feats = self.linlay(word_feats)
        x = torch.cat((doc_feats, word_feats))
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        if mode == 'train':
            mask = data.train_mask
        elif mode == 'val':
            mask = data.val_mask
        elif mode == 'test':
            mask = data.test_mask
        else:
            raise ValueError("Mode '%s' is not supported in forward of graph classifier." % mode)

        x = x[mask]
        return x, data.y[mask]