import json

import torch
from torch.utils.data import Dataset
from transformers import BertJapaneseTokenizer

from src.dependency_llm.util import logger


class dependency_data(Dataset):
    def __init__(
        self, data_path: str, tokenizer: BertJapaneseTokenizer, model_max_length: int
    ) -> None:
        super().__init__()
        # json ファイルを読み込む
        self.data = json.load(open(data_path, "r"))
        self.model_max_length = model_max_length
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id

    # ここで取り出すデータを指定している
    def __getitem__(self, index: int):
        chuk_senetence = self.data[index]["chunk_sentence"]
        head = [int(h) for h in self.data[index]["head"]]
        dep = [int(d) for d in self.data[index]["dep"]]
        assert len(head) == len(dep) == len(chuk_senetence)

        # tokenize and create deps dic
        data = self.create_data(chuk_senetence, head, dep)

        return data

    # この method がないと DataLoader を呼び出す際にエラーを吐かれる
    def __len__(self) -> int:
        return len(self.data)

    def create_data(self, chunk_sentence: list[str], head: list[int], dep: list[int]):
        chunk_tokens, tokenized_sentence = self.tokenize_chunk(chunk_sentence)
        deps_matrix = self.create_deps_matrix(head, dep, chunk_tokens)
        return (deps_matrix, tokenized_sentence)

    def tokenize_chunk(self, chunk_sentence: list[str]):
        chunk_tokens = []
        sentence = "".join(chunk_sentence)
        for idx, chunk in enumerate(chunk_sentence):
            tokens = self.tokenizer.encode(chunk, add_special_tokens=False)
            if idx == 0:
                tokens = [self.cls_token_id] + tokens
            if idx == len(chunk_sentence) - 1:
                tokens = tokens + [self.sep_token_id]
            chunk_tokens.append(tokens)
        tokenized_sentence = self.tokenizer.encode_plus(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=self.model_max_length,
            return_tensors="pt",
        )
        return chunk_tokens, tokenized_sentence

    def create_deps_matrix(self, head, dep, chunk_tokens):
        deps_matrix = torch.zeros(self.model_max_length * self.model_max_length)

        # chunk_tokens内の各要素の最後のtokenから係り受け先の初めのtokenに対しての依存関係があるものとする
        # chunk_tokens内の各要素のそれ以外のtokenに対しては次のtokenに対しての依存関係があるものとする
        scope_dict = self.create_scope_dict(chunk_tokens)

        now_scope_chunk = 0
        for i in range(scope_dict[len(scope_dict) - 1]):
            # 最初は[CLS]なので飛ばす
            if i == 0:
                continue
            # 文章の最後のtoken以降に関しては係り受け関係はないので飛ばす
            if i == scope_dict[len(scope_dict) - 1] - 2:
                break
            if i == scope_dict[now_scope_chunk] - 1:
                # 係り受け関係を表す数値のマッピング処理
                deps_matrix[
                    i * self.model_max_length + scope_dict[dep[now_scope_chunk] - 1]
                ] = 1
                now_scope_chunk += 1
            else:
                # chunk内の擬似的な係り受け関係を表す数値のマッピング処理
                deps_matrix[i * self.model_max_length + i + 1] = 1

        return deps_matrix

    def create_scope_dict(self, chunk_tokens):
        scope_dict = {}
        now_scope = 0
        for i, chunk in enumerate(chunk_tokens):
            now_scope += len(chunk)
            scope_dict[i] = now_scope
        return scope_dict
