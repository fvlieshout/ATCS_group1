from torchtext.data import BucketIterator, Field
from torchtext.vocab import GloVe

ID = Field(sequential=False, include_lengths=False)
TEXT = Field(sequential=True, lower=True, include_lengths=True, batch_first=True)
LABEL = Field(sequential=False, include_lengths=False)


def get_data_splits(dataset, vocab_size, batch_size, emb_dim=300):
    train_split, test_split, val_split = dataset.splits(ID, TEXT, LABEL, val_size=0.1)
    # r8_train, r8_test, r8_val = R8.splits(ID, TEXT, LABEL, val_size=0.1)

    ID.build_vocab(train_split)
    TEXT.build_vocab(train_split, vectors=GloVe(name='840B', dim=emb_dim, max_vectors=vocab_size))
    LABEL.build_vocab(train_split)

    train_iter, test_iter, val_iter = BucketIterator.splits(
        (train_split, test_split, val_split),
        batch_size=batch_size,
        sort=False
    )

    return train_iter, test_iter, val_iter
