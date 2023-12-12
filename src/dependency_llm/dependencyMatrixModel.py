import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertModel

import wandb


class DependencyMatrixModel(nn.Module):
    def __init__(
        self, model_name: str, add_layer_size: int, max_length=512, dropout=0.2
    ) -> None:
        super(DependencyMatrixModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.freeze(self.bert)

        self.max_length = max_length
        hidden_size = self.bert.config.hidden_size

        self.add_layer_size = add_layer_size

        # self.linear_layer = torch.nn.Linear(
        #     self.max_length * hidden_size * self.add_layer_size,
        #     self.max_length * hidden_size * self.add_layer_size,
        # )
        # self.relu = torch.nn.ReLU()
        # self.dropout = torch.nn.Dropout(dropout)
        #
        self.last_layer = torch.nn.Linear(self.max_length, max_length * max_length)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        output = self.bert(
            input_ids, attention_mask=attention_mask, output_hidden_states=True
        )

        # sequence_output = torch.cat(
        #     [
        #         output["hidden_states"][-1 * i]
        #         for i in range(1, self.add_layer_size + 1)
        #     ],
        #     dim=2,
        # ).to(
        #     "cuda"
        # )  # (batch_size, seq_length, hidden_size * add_layer_size)

        # (batch_size, seq_length, hidden_size)
        sequence_output = output["hidden_states"][-1]
        # (batch_size, seq_length)になるようにhidden_sizeを足し合わせて圧縮
        sequence_output = torch.sum(sequence_output, dim=2)
        sequence_output = self.last_layer(sequence_output)

        return sequence_output

    # embedding層のパラメータなどを固定できる
    def freeze(self, model: torch.nn.Module):
        """
        Freezes module's parameters.
        """
        for parameter in model.parameters():
            parameter.requires_grad = False
