import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from transformers.data.data_collator import default_data_collator
import torch
from torch import nn
from transformers import Trainer

DEFAULT_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class RobertaTrainer(Trainer):

    def __init__(self, train_set, test_set, val_set, model, roberta_hid_dim, num_classes, args):
        super().__init__(model=model, train_dataset=train_set, eval_dataset=test_set, args=args)

        # TODO: Check how exactly validation/test set are used in trainer
        # self.val_set = val_set

        self.classifier = nn.Sequential(
            nn.Linear(roberta_hid_dim, 256),
            nn.Linear(256, num_classes)
        ).to(DEFAULT_DEVICE)

        self.loss_module = nn.CrossEntropyLoss()

    def training_step(self, model, batch):
        labels = batch['labels'].to(DEFAULT_DEVICE)
        inputs = batch['input_ids'].to(DEFAULT_DEVICE)

        # returns a tuple of torch.FloatTensor comprising various elements depending on the (RobertaConfig) and inputs.
        roberta_output = model(inputs)

        # b_size x seq_len x hid_size
        last_hidden_state = roberta_output[0]

        # b_size x hid_size
        cls_token_state = roberta_output[1]

        predictions = self.classifier(cls_token_state)

        loss = self.loss_module(predictions, labels)

        metrics = {'train_loss': loss, 'train_acc': self.accuracy(predictions, labels)}
        # noinspection PyUnresolvedReferences
        self.log(metrics)

        return loss

    def prediction_step(self, batch, batch_idx):
        # TODO
        pass

    @staticmethod
    def accuracy(predictions, labels):
        # noinspection PyUnresolvedReferences
        return (labels == predictions.argmax(dim=-1)).float().mean()

    # def get_train_dataloader(self, train_dataset):
    #     return DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=default_data_collator)

    # def get_eval_dataloader(self):
    #     return DataLoader(self.val_set, batch_size=2, shuffle=True, collate_fn=default_data_collator)

    # def get_test_dataloader(self):
    #     return DataLoader(self.test_set, batch_size=2, shuffle=True, collate_fn=default_data_collator)
