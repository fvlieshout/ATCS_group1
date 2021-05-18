from nltk import word_tokenize

import torch
from torchtext.vocab import GloVe

from data_prep.graph_dataset import GraphDataset


class GloveGraphDataset(GraphDataset):
    """
    Text Dataset used by the pure graph model.
    """

    def _generate_features(self):
        """
        Generates node features.

        Returns:
            note_feats (Tensor): Tensor of all node embeddings.
        """
        self.doc_dim = 10000
        self.word_dim = 300

        features_docs = []
        features_words = []
        glove = GloVe(name='840B', dim=300, max_vectors=10000)

        print('Generating document node features')
        for text in self._raw_texts:
            tokens = set(word_tokenize(text.lower()))
            inds = torch.tensor([glove.stoi[token] for token in tokens if token in glove.stoi])
            # Use only 10k most common tokens
            inds = inds[inds < self.doc_dim]
            doc_feat = torch.zeros(self.doc_dim)
            if len(inds) > 0:
                doc_feat[inds] = 1
            features_docs.append(doc_feat)
        features_docs = torch.stack(features_docs).to(self._device)

        print('Generating word node features')
        for token in self._tokens:
            features_words.append(glove[token])
        features_words = torch.stack(features_words).to(self._device)

        return features_docs, features_words
