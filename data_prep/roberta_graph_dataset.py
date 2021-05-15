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
    def __init__(self, corpus, device, *args, **kwargs):
        super().__init__(device, corpus)

        print('Generating feature matrix')
        self.data.doc_features, self.data.word_features = self.generate_features(self.tokens, self.raw_text)

    def preprocess(self, raw_texts):
        """
        Preprocesses the corpus.

        Args:
            raw_texts (List): List of raw untokenized texts of all documents

        Returns:
            tokenized_text (List): List of tokenized documents texts.
            tokens (List): List of all tokens.
        """
        tokenized_text = [word_tokenize(text.lower()) for text in raw_texts]
        tokens = sorted(list(set([token for text in tokenized_text for token in text])))
        tokenized_text, tokens

    def generate_features(self, tokens, raw_text):
        """
        Generates node features.

        Args:
            tokens (List): List of all tokens in the corpus.
            raw_text ([type]): List of raw document texts.

        Returns:
            features_docs (Tensor): Tensor of document node embeddings.
            features_words (Tensor): Tensor of token node embeddings.
        """
        features_docs = []
        features_words = []
        doc_embedder = RobertaModel.from_pretrained('roberta-base')
        token_embedder = GloVe(name='840B', dim=300)

        for text in raw_text:
            encodings = self.roberta_tokenizer.encode(text, truncation=True, padding=False)
            encodings = torch.tensor(encodings, dtype=torch.long).unsqueeze()
            embed_doc = doc_embedder(encodings)[1]
            features_docs.append(embed_doc.squeeze())
        features_docs = torch.stack(features_docs)
        features_docs = features_docs.to(self.device)

        for token in tokens:
            embed_token = token_embedder[token]
            features_words.append(embed_token)
        features_words = torch.stack(features_words)
        features_words = features_words.to(self.device)

        return features_docs, features_words
