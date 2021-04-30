import argparse
import os

from transformers import RobertaConfig, RobertaModel
from transformers import Trainer, TrainingArguments

from data import get_data_splits
from datasets.reuters_text import R8

LOG_PATH = "./logs/"


def train(seed, epochs, b_size, l_rate, vocab_size):
    os.makedirs(LOG_PATH, exist_ok=True)

    configuration = RobertaConfig(
        vocab_size=vocab_size,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )

    # Initializing a model from the configuration
    model = RobertaModel(configuration)

    # # TODO: put the real dataset here
    # # should be sampled sub graphs
    # # raw feature embeddings of nodes in subgraph batch
    # train_dataset = torch.utils.data.Dataset()
    #
    # # TODO: implement this:  mini-batches from the subgraph datasets
    # data_collator = None

    train_iter, test_iter, val_iter = get_data_splits(R8, vocab_size, b_size)

    # noinspection PyTypeChecker
    training_args = TrainingArguments(
        output_dir=LOG_PATH,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_gpu_train_batch_size=b_size,
        # save_total_limit=2,  will limit the total amount of checkpoints. Deletes the older checkpoints in output_dir
        learning_rate=l_rate,
        evaluation_strategy='epoch',  # evaluate at the end of each epoch
        seed=seed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        # data_collator=data_collator,  # function for forming batch from list of elements of train_dataset/eval_dataset
        train_dataset=train_iter,
        prediction_loss_only=True,
    )

    trainer.train()

    # trainer.save_model("./models/roberta-retrained")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # TRAINING PARAMETERS

    parser.add_argument('--epochs', dest='epochs', type=int, default=20)
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=4)
    parser.add_argument('--lr', dest='l_rate', type=float, default=0.1)

    # CONFIGURATION

    parser.add_argument('--seed', dest='seed', type=int, default=1234)
    parser.add_argument('--vocab-size', dest='vocab_size', type=int, default=10000)

    params = vars(parser.parse_args())

    train(params['seed'],
          params['epochs'],
          params["batch_size"],
          params["l_rate"],
          params["vocab_size"])
