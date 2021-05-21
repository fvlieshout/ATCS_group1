from collections import defaultdict

import nltk

nltk.download('punkt')
from nltk import word_tokenize
import torch
from torchtext.vocab import GloVe
from transformers import RobertaModel

from data_prep.graph_dataset import GraphDataset
from data_prep.roberta_dataset import RobertaDataset
from models.roberta_encoder import *


class RobertaGraphDataset(GraphDataset):
    """
    Text Dataset used by the Roberta graph model.
    """

    def __init__(self, corpus, roberta_model):
        # this needs to be set BEFORE the constructor call, because the stuff in the super constructor needs this
        self.roberta_checkpoint = roberta_model

        super(RobertaGraphDataset, self).__init__(corpus)

    def _generate_features(self):
        """
        Generates node features.
        Returns:
            features_docs (Tensor): Tensor of document node embeddings.
            features_words (Tensor): Tensor of token node embeddings.
        """
        features_docs, features_words = [], []

        doc_embedder = self.get_doc_embedder().to(self._device)
        token_embedder = GloVe(name='840B', dim=300)

        with torch.no_grad():
            print('Generating document node features')
            batch_size = 64
            num_batches = int(len(self._raw_texts) / batch_size) + 1
            for i in range(num_batches):
                max_docs = len(self._raw_texts) if i == num_batches - 1 else (i + 1) * batch_size
                docs = self._raw_texts[i * batch_size:max_docs]
                encodings = self._tokenizer(docs, truncation=True, padding=True)
                items = {k: torch.tensor(v, dtype=torch.long, device=self._device) for k, v in encodings.items()}
                if self.roberta_checkpoint is None:
                    _, out = doc_embedder(items['input_ids'])
                else:
                    out, _ = doc_embedder(items)
                features_docs.append(out)
            features_docs = torch.cat(features_docs, dim=0).to(self._device)

            print('Generating word node features')
            for token in self._tokens:
                embed_token = token_embedder[token]
                features_words.append(embed_token)
            features_words = torch.stack(features_words).to(self._device)

        return features_docs, features_words

    def get_doc_embedder(self):
        """
        Either loads a pretrained/fine-tuned roberta encoder or a non-fine-tuned one and returns it.
        Returns:
            encoder (RobertaModel): Roberta encoder.
        """

        if self.roberta_checkpoint is None:
            # use off-the-shelve pretrained Roberta Model
            embedder = RobertaModel.from_pretrained('roberta-base').to(self._device)
        else:
            # use finetunes Roberta Model
            embedder = RobertaEncoder()
            embedder.load_state_dict(self.get_encoder_state_dict())
        return embedder

    def get_encoder_state_dict(self):
        encoder_state_dict = {}
        for layer_key, param in torch.load(self.roberta_checkpoint)['state_dict'].items():
            if layer_key.startswith("model"):
                new_key = layer_key[layer_key.index(".") + 1:]
                encoder_state_dict[new_key] = param
        return encoder_state_dict
