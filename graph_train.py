import os
import argparse
import torch
from collections import defaultdict
import random

import nltk
import torch_geometric.data as geom_data
# nltk.download('reuters')
# from nltk.corpus import reuters
from datasets.reuters_graph_datasets import R8, R52
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


from datasets.graph_utils import get_PMI, tf_idf_mtx

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class Net(torch.nn.Module):
    def __init__(self, num_nodes):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_nodes, 200)
        self.conv2 = GCNConv(200, 8)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

class Graph_model(pl.LightningModule):
    def __init__(self, num_nodes):
        super().__init__()
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.model = Net(num_nodes).to(device)
        self.save_hyperparameters()
    
    def forward(self, data, mode):
        out = self.model(data)
        if mode=='train':
            mask = data.train_mask
        elif mode=='val':
            mask = data.val_mask
        elif mode=='test':
            mask = data.test_mask
        loss = F.cross_entropy(out[mask], data.y[mask])
        # loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        class_predictions = torch.argmax(out, dim=1)
        correct = (class_predictions[mask] == data.y[mask]).sum().item()
        accuracy = correct / mask.sum()
        # if math.isnan(loss.item()):
        #     print()
        return loss, accuracy
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return self.optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode='train')
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode='val')
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode='test')
        self.log("test_loss", loss)
        self.log("test_acc", acc)

class GenerateCallback(pl.Callback):
    def __init__(self):
        """
        Args:
            gamma (float, optional): [description]. Defaults to 0.2.
        """
        super().__init__()

    def on_epoch_end(self, trainer, pl_module):
        """
        This function is called after every epoch.
        In this function, the dev accuracy is checked and the learning rate is adjusted if dev accuracy has decreased
        """
        val_acc = trainer.callback_metrics.get('val_acc').item()
        print('Epoch [' + str(trainer.current_epoch+1) + ']; train_loss: ' + str(trainer.callback_metrics.get('train_loss').item())
                     + '; val_loss: ' + str(trainer.callback_metrics.get('val_loss').item()) + ': val_acc: ' + str(val_acc))

def train_model(args):
    os.makedirs(args.log_dir, exist_ok=True)

    if args.dataset=='r8':
        dataset = R8(device)
    elif args.dataset=='r52':
        dataset = R52(device)

    # data = data_object.data.to(device)
    graph_data_loader = geom_data.DataLoader(dataset, batch_size=1)

    # Create a PyTorch Lightning trainer with the generation callback
    gen_callback = GenerateCallback()
    trainer = pl.Trainer(default_root_dir=args.log_dir,
                         checkpoint_callback=ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss"),
                         gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs=args.epochs,
                         callbacks=[gen_callback],
                         progress_bar_refresh_rate=1 if args.progress_bar else 0) 
    trainer.logger._default_hp_metric = None 

    # Create model
    pl.seed_everything(args.seed)
    model = Graph_model(len(dataset.iton))
    
    # optionally resume from a checkpoint
    if args.resume != None:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint %s' % args.resume)
            model = NLI_model.load_from_checkpoint(args.resume)
            print('=> loaded checkpoint %s' % args.resume)
        else:
            print('=> no checkpoint found at %s' % args.resume)

    #Training
    trainer.fit(model, graph_data_loader, graph_data_loader)

if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--dataset', default='r8', type=str,
                        help='What dataset to use',
                        choices=['r8', 'r52'])
    parser.add_argument('--embedding_dim', default=300, type=int,
                        help='Embedding dimensionalities')
    parser.add_argument('--classifier_hidden', default=512, type=int,
                        help='Hidden dimensionalities to use inside the classifier')
    parser.add_argument('--LSTM_hidden', default=2048, type=int,
                        help='Hidden dimensionalities to use inside the LSTM model')
    parser.add_argument('--output_dim', default=3, type=int,
                        help='Number of classes')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--LSTM_dropout', default=0.0, type=float,
                        help='Dropout to use in the LSTM models')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to use in the data loaders. To have a truly deterministic run, this has to be 0. ' + \
                             'For your assignment report, you can use multiple workers (e.g. 4) and do not have to set it to 0.')
    parser.add_argument('--log_dir', default='graph_logs', type=str,
                        help='Directory where the PyTorch Lightning logs should be created.')
    parser.add_argument('--progress_bar', action='store_true',
                        help=('Use a progress bar indicator for interactive experimentation. '
                              'Not to be used in conjuction with SLURM jobs'))

    args = parser.parse_args()
    train_model(args)
