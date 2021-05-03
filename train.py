import argparse
import os
from datasets.reuters_text import R8
from model import RobertaTrainer, DEFAULT_DEVICE
import torch
from torch.utils.data import DataLoader
from torchtext.data import Field
from transformers import RobertaConfig, RobertaModel
from transformers import RobertaTokenizerFast
from transformers import TrainingArguments
from transformers import Trainer, TrainingArguments
from transformers.data.data_collator import default_data_collator

from datasets.reuters_text import R8
from model import RobertaTrainer, DEFAULT_DEVICE

LOG_PATH = "./logs/"

SUPPORTED_MODELS = ['baseline']


def train(model, seed, epochs, b_size, l_rate, vocab_size, dataset=R8):
    os.makedirs(LOG_PATH, exist_ok=True)

    if model == 'baseline':
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

        configuration = RobertaConfig(
            vocab_size=tokenizer.vocab_size,
            max_position_embeddings=514,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1,
        )

        model = RobertaModel(configuration)

        train_dataset, test_dataset, val_dataset = dataset.splits(tokenizer, val_size=0.1)
        num_classes = 8 if dataset == R8 else 52

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
        seed=seed,
        place_model_on_device=True
    )

    trainer = RobertaTrainer(train_dataset, test_dataset, val_dataset,
                             model=model.to(DEFAULT_DEVICE),
                             roberta_hid_dim=768,
                             num_classes=num_classes,
                             args=training_args)

    trainer.train()

    # trainer.save_model("./models/roberta-retrained")


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
