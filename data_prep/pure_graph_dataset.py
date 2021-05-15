import torch

from data_prep.graph_dataset import GraphDataset


class PureGraphDataset(GraphDataset):
    """
    Text Dataset used by the pure graph model.
    """
    def __init__(self, corpus, device):
        super().__init__(corpus, device)
        print('Generating feature matrix')
        self._data.x = self._generate_features(self._tokens, self._raw_texts)

    def _preprocess(self, raw_texts):
        """
        Preprocesses the corpus.

        Args:
            raw_texts (List): List of raw untokenized texts of all documents

        Returns:
            tokenized_text (List): List of tokenized documents texts.
            tokens (List): List of all tokens.
        """
        # Run Roberta tokenizer on corpus
        tokenized_text = [self._tokenizer.tokenize(text.lower()) for text in raw_texts]
        tokens = [word for word, _ in sorted(self._tokenizer.vocab.items(), key=lambda item: item[1])]
        return tokenized_text, tokens

    def _generate_features(self, tokens, raw_texts):
        """
        Generates node features.

        Args:
            tokens (List): List of all tokens in the corpus.
            raw_texts ([type]): List of raw document texts.

        Returns:
            note_feats (Tensor): Tensor of all node embeddings.
        """
        features_docs = []

        print('Generating document node features')
        for text in raw_texts:
            encodings = self._tokenizer.encode(text, truncation=True, padding=False)
            encodings = torch.tensor(encodings)
            embed_doc = torch.zeros(self._tokenizer.vocab_size)
            embed_doc[encodings] = 1
            features_docs.append(embed_doc)
        features_docs = torch.stack(features_docs)
        features_docs = features_docs.to(self._device)

        print('Generating word node features')
        features_words = torch.eye(len(tokens), device=self._device).float()

        node_feats = torch.cat((features_docs, features_words))
        return node_feats
