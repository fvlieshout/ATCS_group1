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

    def __init__(self, corpus):
        super().__init__(corpus)
        self._data.doc_features, self._data.word_features = self._generate_features()

    def _preprocess(self, lower_threshold=4, upper_threshold=50):
        """
        Preprocesses the corpus.

        Returns:
            tokenized_text (List): List of tokenized documents texts.
            tokens (List): List of all tokens.
        """
        tokenized_text = [word_tokenize(text.lower()) for text in self._raw_texts]
        counter = defaultdict(lambda: 0)
        for text in tokenized_text:
            for token in set(text):
                counter[token] += 1

        tokenized_text = [
            [token for token in text if counter[token] >= lower_threshold and counter[token] < upper_threshold]
            for text in tokenized_text]
        tokens = list(set([token for text in tokenized_text for token in text]))
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
        doc_embedder = RobertaModel.from_pretrained('roberta-base').to(self._device)
        token_embedder = GloVe(name='840B', dim=300)

        with torch.no_grad():
            print('Generating document node features')
            encodings = self._tokenizer(self._raw_texts, truncation=True, padding=True)['input_ids']
            encodings = torch.tensor(encodings, dtype=torch.long, device=self._device)
            features_docs = doc_embedder(encodings)[1]

            print('Generating word node features')
            for token in self._tokens:
                embed_token = token_embedder[token]
                features_words.append(embed_token)
        features_words = torch.stack(features_words).to(self._device)

        return features_docs, features_words
