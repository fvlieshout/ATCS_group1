from torch import nn
from transformers import RobertaModel


class RobertaEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # transformer_config = model_hparams['transformer_config']
        self.model = RobertaModel.from_pretrained('roberta-base')

        # this model is in eval per default, we want to fine-tune it but only the top layers
        self.model.train()

        # only freezing the encoder parameters / weights of the head layers
        for param in self.model.base_model.parameters():
            param.requires_grad = False

    # noinspection PyUnusedLocal
    def forward(self, batch, **kwargs):
        """
        Pushes the data through the roberta model and returns the last hidden state (CLS token) from the output.
        Args:
            batch (dict): Dictionary of lists/arrays/tensors returned by the encode method of huggingface ('input_ids',
            'attention_mask', etc.).
            **kwargs (Map): Will catch additional unused attributes.
        Returns
            out (Tensor): CLS hidden state as tensor.
            labels (List): List of int containing the labels of the batch.
        """

        inputs, attention_mask = batch['input_ids'], batch['attention_mask']
        # returns a tuple of torch.FloatTensor comprising various elements depending on the (RobertaConfig) and inputs.
        hidden_states = self.model(inputs, attention_mask)

        # b_size x hid_size
        out = hidden_states[1]
        return out, batch['labels']