import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchtext.data import Field
from transformers import RobertaConfig, RobertaModel
from transformers import RobertaTokenizerFast
from transformers import Trainer, TrainingArguments

from datasets.reuters_text import R8

LOG_PATH = "./logs/"

SUPPORTED_MODELS = ['baseline']

ID = Field(sequential=False, include_lengths=False)
TEXT = Field(sequential=True, lower=True, include_lengths=True, batch_first=True)
LABEL = Field(sequential=False, include_lengths=False)


def train(model, seed, epochs, b_size, l_rate, vocab_size):
    os.makedirs(LOG_PATH, exist_ok=True)

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    configuration = RobertaConfig(
        vocab_size=vocab_size,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )

    if model == 'baseline':
        # Initializing a model from the configuration
        model = RobertaModel(configuration)

        train_dataset, test_dataset, val_dataset = R8.splits(ID, TEXT, LABEL, val_size=0.1)

        # random stuff, trying to get the dataset into the right format
        train_samples = [' '.join(exp.text) for exp in train_dataset.examples]
        val_samples = [' '.join(exp.text) for exp in val_dataset.examples]
        test_samples = [' '.join(exp.text) for exp in test_dataset.examples]

        # actually, all data points should be put into the tokenizer once so that we have the full vocab; splitting into datasets should happen after??
        train_encodings = tokenizer(train_samples, truncation=True, padding=True)
        val_encodings = tokenizer(val_samples, truncation=True, padding=True)
        test_encodings = tokenizer(test_samples, truncation=True, padding=True)

    else:
        raise ValueError("Model type '%s' is not supported." % model)

    # noinspection PyTypeChecker
    training_args = TrainingArguments(
        output_dir=LOG_PATH,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=b_size,
        # save_total_limit=2,  will limit the total amount of checkpoints. Deletes the older checkpoints in output_dir
        learning_rate=l_rate,
        evaluation_strategy='epoch',  # evaluate at the end of each epoch
        seed=seed
    )

    trainer = RobertaTrainer(train_encodings, test_encodings, val_encodings, model=model, args=training_args)
    # data_collator=data_collator,  # function for forming batch from list of elements of train_dataset/eval_dataset

    trainer.train()

    # trainer.save_model("./models/roberta-retrained")


def collate_examples(batch):
    """

    :param batch: List of torchtext.data.example.Example
    :return:
    """
    print(batch[0].text)

    # text contains sentence and sentence length
    texts = [encoding.tokens for encoding in batch]
    texts = torch.LongTensor(texts)

    labels = [example.label for example in batch]
    labels = torch.LongTensor(labels)

    return [texts, labels]


class RobertaTrainer(Trainer):

    def __init__(self, train_encodings, test_encodings, val_encodings, model, args):
        super().__init__(model=model, args=args)

        self.train_encodings = train_encodings
        self.test_encodings = test_encodings
        self.val_encodings = val_encodings

    def get_train_dataloader(self):
        return DataLoader(
            self.train_encodings,
            # batch_size=batch_size  # Trains with this batch size.
            collate_fn=collate_examples
        )

    def get_eval_dataloader(self):
        return DataLoader(
            self.val_encodings,
            # batch_size=batch_size  # Trains with this batch size.
            collate_fn=collate_examples
        )

    def get_test_dataloader(self):
        return DataLoader(
            self.test_encodings,
            # batch_size=batch_size  # Trains with this batch size.
            collate_fn=collate_examples
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # TRAINING PARAMETERS

    parser.add_argument('--epochs', dest='epochs', type=int, default=20)
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=4)
    parser.add_argument('--lr', dest='l_rate', type=float, default=0.1)

    # CONFIGURATION

    parser.add_argument('--model', dest='model', default='baseline', choices=SUPPORTED_MODELS,
                        help='Select the model you want to use.')
    parser.add_argument('--seed', dest='seed', type=int, default=1234)
    parser.add_argument('--vocab-size', dest='vocab_size', type=int, default=10000)

    params = vars(parser.parse_args())

    train(params['model'],
          params['seed'],
          params['epochs'],
          params["batch_size"],
          params["l_rate"],
          params["vocab_size"])
