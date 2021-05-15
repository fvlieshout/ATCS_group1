import nltk
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
    def __init__(self, corpus, device):
        super().__init__(corpus, device)
        self._data.doc_features, self._data.word_features = self._generate_features()

    def _preprocess(self):
        """
        Preprocesses the corpus.

        Returns:
            tokenized_text (List): List of tokenized documents texts.
            tokens (List): List of all tokens.
        """
        tokenized_text = [word_tokenize(text.lower()) for text in self._raw_texts]
        tokens = sorted(list(set([token for text in tokenized_text for token in text])))
        return tokenized_text, tokens

    def _generate_features(self):
        """
        Generates node features.

        Returns:
            features_docs (Tensor): Tensor of document node embeddings.
            features_words (Tensor): Tensor of token node embeddings.
        """
        features_docs = []
        features_words = []
        doc_embedder = RobertaModel.from_pretrained('roberta-base')
        token_embedder = GloVe(name='840B', dim=300)

        print('Generating document node features')
        for text in self._raw_texts:
            encodings = self._tokenizer.encode(text, truncation=True, padding=False)
            encodings = torch.tensor(encodings, dtype=torch.long).unsqueeze(0)
            embed_doc = doc_embedder(encodings)[1]
            features_docs.append(embed_doc.squeeze())
        features_docs = torch.stack(features_docs)
        features_docs = features_docs.to(self._device)

        print('Generating word node features')
        for token in self._tokens:
            embed_token = token_embedder[token]
            features_words.append(embed_token)
        features_words = torch.stack(features_words)
        features_words = features_words.to(self._device)

        return features_docs, features_words
