import torch
import torch.nn as nn
from transformers import BertModel
from util import logger


class baseModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        scope_layer_size: int,
        max_length=512,
        num_labels=2,
        dropout=0.2,
        hidden_size=1000,
        freeze_bert=True,
    ) -> None:
        super(baseModel, self).__init__()
        logger.info("Loading Base model...")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        scope: torch.Tensor,
    ):
        batch_size = input_ids.shape[0]

        score = []
        for i in range(batch_size):
            if scope[i][0] + 1 == scope[i][1]:
                score.append([0.0, 1.0])
            else:
                score.append([1.0, 0.0])

        score = torch.tensor(
            score,
        ).to("cuda")

        return score  # (batch_size, num_labels)
