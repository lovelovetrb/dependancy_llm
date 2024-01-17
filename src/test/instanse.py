import json

import yaml
from transformers import BertJapaneseTokenizer

from src.dependency_llm.dependency_data import dependency_data


class TestInstance:
    def __init__(self) -> None:
        print("test_instance")
        data_path = "src/dependency_llm/data/test.json"
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(
            "cl-tohoku/bert-base-japanese"
        )
        self.model_max_length = self.tokenizer.max_model_input_sizes[
            "cl-tohoku/bert-base-japanese"
        ]

        self.dataset = dependency_data(
            data_path=data_path,
            model_max_length=self.model_max_length,
            tokenizer=self.tokenizer,
        )

        self.deps_data = json.load(open(data_path, "r"))
