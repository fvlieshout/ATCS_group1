import nltk
from collections import defaultdict

nltk.download('punkt')
from nltk import word_tokenize
import torch
from torchtext.vocab import GloVe
from transformers import RobertaModel

from data_prep.graph_dataset import GraphDataset


class RobertaGraphDataset(GraphDataset):
    """
    Text Dataset used by the Roberta graph model.
    """

    def _generate_features(self):
        """
        Generates node features.
        Returns:
            features_docs (Tensor): Tensor of document node embeddings.
            features_words (Tensor): Tensor of token node embeddings.
        """
        features_docs = []
        features_words = []
        doc_embedder = RobertaModel.from_pretrained('roberta-base').to(self._device)
        token_embedder = GloVe(name='840B', dim=300)

        with torch.no_grad():
            print('Generating document node features')
            batch_size = 64
            num_batches = int(len(self._raw_texts) / batch_size) + 1
            for i in range(num_batches):
                if i == num_batches - 1:
                    docs = self._raw_texts[i * batch_size:]
                else:
                    docs = self._raw_texts[i * batch_size:(i + 1) * batch_size]
                encoding = self._tokenizer(docs, truncation=True, padding=True)['input_ids']
                encoding = torch.tensor(encoding, dtype=torch.long, device=self._device)
                encoding = doc_embedder(encoding)[1]
                features_docs.append(encoding)
            features_docs = torch.cat(features_docs, dim=0).to(self._device)

            print('Generating word node features')
            for token in self._tokens:
                embed_token = token_embedder[token]
                features_words.append(embed_token)
            features_words = torch.stack(features_words).to(self._device)

        return features_docs, features_words
