import torch
import torch.nn as nn
from transformers import BertModel
from util import logger


class DependencyMatrixModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        scope_layer_size: int,
        max_length=512,
        dropout=0.2,
        hidden_size=1000,
        freeze_bert=True,
    ) -> None:
        super(DependencyMatrixModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        if freeze_bert is True:
            logger.info("Freezing BERT parameters...")
            self.freeze(self.bert)

        self.max_length = max_length
        bert_output_dim = self.bert.config.hidden_size

        self.scope_layer_size = scope_layer_size

        self.linear_layer = torch.nn.Linear(
            bert_output_dim * 2 * self.scope_layer_size,
            hidden_size,
        )
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.last_layer = torch.nn.Linear(hidden_size, 1)

    def forward(
        self,
        dep_data: dict,
    ):
        input_ids = dep_data[0][0]["token"]["input_ids"].squeeze(0).to("cuda")
        attention_mask = dep_data[0][0]["token"]["attention_mask"].squeeze(0).to("cuda")
        scope = dep_data[0][0]["scope"]

        output = self.bert(
            input_ids, attention_mask=attention_mask, output_hidden_states=True
        )

        bert_hid_stats_n_layer = torch.cat(
            [
                output["hidden_states"][-1 * i]
                for i in range(1, self.scope_layer_size + 1)
            ],
            dim=2,
        ).to(
            "cuda"
        )  # (batch_size, seq_length, hidden_size * scope_layer_size)

        # (batch_size, seq_length, hidden_size * scope_layer_size) -> (batch_size, 2 , hidden_size * scope_layer_size)
        scope_hid_stats = torch.stack(
            [bert_hid_stats_n_layer[i, scope[i], :] for i in range(len(scope))],
            dim=0,
        )  # (batch_size, 2 , hidden_size * scope_layer_size)

        # (batch_size, 2 , hidden_size * scope_layer_size) -> (batch_size, 2 * hidden_size * scope_layer_size)
        scope_hid_stats = scope_hid_stats.reshape(len(scope), -1)

        assert scope_hid_stats.shape == (
            input_ids.shape[0],  # batch_size
            2 * self.bert.config.hidden_size * self.scope_layer_size,
        )

        sequence_output = self.linear_layer(scope_hid_stats)
        sequence_output = self.relu(sequence_output)
        sequence_output = self.dropout(sequence_output)

        score = self.last_layer(sequence_output)  # (batch_size, 2)
        return score

    # embedding層のパラメータなどを固定できる
    def freeze(self, model: torch.nn.Module):
        """
        Freezes module's parameters.
        """
        for parameter in model.parameters():
            parameter.requires_grad = False

    def knockout(self, knockout_layer_head: list[list[int]]):
        """
        Knockout layer.
        """
        logger.info(f"Knockout layer... : {len(knockout_layer_head)}")
        for layer_idx, head_idx in knockout_layer_head:
            logger.info(f"Knockout layer {layer_idx} head {head_idx}...")
            # 自己注意の重みを取得
            attention = self.bert.encoder.layer[layer_idx - 1].attention.self

            # キー、クエリ、バリューの重みとバイアスをゼロに設定
            start = (head_idx - 1) * attention.attention_head_size
            end = head_idx * attention.attention_head_size

            attention.query.weight.data[:, start:end] = 0
            attention.key.weight.data[:, start:end] = 0
            attention.value.weight.data[:, start:end] = 0

            attention.query.bias.data[start:end] = 0
            attention.key.bias.data[start:end] = 0
            attention.value.bias.data[start:end] = 0
