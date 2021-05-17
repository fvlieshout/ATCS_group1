import torch

from data_prep.graph_dataset import GraphDataset


class PureGraphDataset(GraphDataset):
    """
    Text Dataset used by the pure graph model.
    """

    def __init__(self, corpus):
        super().__init__(corpus)
        print('Generating feature matrix')
        self._data.x = self._generate_features()

    def _preprocess(self):
        """
        Preprocesses the corpus.

        Returns:
            tokenized_text (List): List of tokenized documents texts.
            tokens (List): List of all tokens.
        """
        # Run Roberta tokenizer on corpus
        tokenized_text = [self._tokenizer.tokenize(text.lower()) for text in self._raw_texts]
        tokens = [word for word, _ in sorted(self._tokenizer.vocab.items(), key=lambda item: item[1])]
        return tokenized_text, tokens

    def _generate_features(self):
        """
        Generates node features.

        Returns:
            note_feats (Tensor): Tensor of all node embeddings.
        """
        features_docs = []

        print('Generating document node features')
        for text in self._raw_texts:
            encodings = self._tokenizer.encode(text, truncation=True, padding=False)
            encodings = torch.tensor(encodings)
            embed_doc = torch.zeros(self._tokenizer.vocab_size)
            embed_doc[encodings] = 1
            features_docs.append(embed_doc)
        features_docs = torch.stack(features_docs).to(self._device)

        print('Generating word node features')
        features_words = torch.eye(len(self._tokens), device=self._device).float()

        node_feats = torch.cat((features_docs, features_words))
        return node_feats
