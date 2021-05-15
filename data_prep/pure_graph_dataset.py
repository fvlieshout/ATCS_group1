import torch

from data_prep.graph_dataset import GraphDataset


class PureGraphDataset(GraphDataset):
    """
    Text Dataset used by the pure graph model.
    """
    def __init__(self, corpus, device, *args, **kwargs):
        super().__init__(device, corpus)
        self.data.x = self.generate_features(self.tokens, self.raw_text)

    def preprocess(self, raw_texts):
        """
        Preprocesses the corpus.

        Args:
            raw_texts (List): List of raw untokenized texts of all documents

        Returns:
            tokenized_text (List): List of tokenized documents texts.
            tokens (List): List of all tokens.
        """
        # Run Roberta tokenizer on corpus
        tokenized_text = [self.roberta_tokenizer.tokenize(text.lower()) for text in raw_texts]
        tokens = [word for word, _ in sorted(self.roberta_tokenizer.vocab.items(), key=lambda item: item[1])]
        return tokenized_text, tokens

    def generate_features(self, tokens, raw_text):
        """
        Generates node features.

        Args:
            tokens (List): List of all tokens in the corpus.
            raw_text ([type]): List of raw document texts.

        Returns:
            note_feats (Tensor): Tensor of all node embeddings.
        """
        features_docs = []

        for text in raw_text:
            encodings = self.roberta_tokenizer.encode(text, truncation=True, padding=False)
            encodings = torch.tensor(encodings)
            embed_doc = torch.zeros(self.tokenizer.vocab_size)
            embed_doc[encodings] = 1
            features_docs.append(embed_doc)
        features_docs = torch.stack(features_docs)
        features_docs = features_docs.to(self.device)

        # Token features are an Identity matrix
        features_words = torch.eye(len(tokens), device=self.device).float()

        node_feats = torch.cat((features_docs, features_words))
        return node_feats
