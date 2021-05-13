#!/usr/bin/env python3
from data_prep.dataset import HuggingFaceDatasetText


class AGNewsText(HuggingFaceDatasetText):
    """
    Wrapper for the AGNews dataset.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def splits(cls, *args, **kwargs):
        return super().splits(dataset_name="ag_news", *args, **kwargs)


if __name__ == "__main__":
    from transformers import RobertaTokenizer
    from torch.utils.data import DataLoader

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    train_set, test_set, val_set = AGNewsText.splits(tokenizer)
    print("train size=", len(train_set))
    print("test size=", len(test_set))
    print("val size=", len(val_set))

    train_loader = DataLoader(train_set, batch_size=2, shuffle=True, collate_fn=train_set.get_collate_fn())

    for data_batch in train_loader:
        print(data_batch)
        input_ids = data_batch["input_ids"]
        attention_mask = data_batch["attention_mask"]
        d_labels = data_batch["labels"]
        print(input_ids)
        print(attention_mask)
        print(d_labels)
        break
