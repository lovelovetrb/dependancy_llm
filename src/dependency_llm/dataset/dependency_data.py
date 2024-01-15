import json

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertJapaneseTokenizer
from util import logger

# from src.dependency_llm.util import logger


class dependency_data(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: BertJapaneseTokenizer,
        model_max_length: int,
    ) -> None:
        super().__init__()
        # json ファイルを読み込む
        self.data = json.load(open(data_path, "r"))
        # self.data = self.data[:100]
        self.model_max_length = model_max_length
        self.tokenizer = tokenizer

        self.pad_token_id = tokenizer.pad_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id

        self.dep_data = []

        self.pos_outside_neighbor_num = 0
        self.pos_outside_num = 0
        self.pos_outside_distant_num = 0
        self.pos_inside_neighbor_num = 0
        self.neg_neighbor_num = 0
        self.neg_distant_num = 0

        # DEBUG
        index = 0
        for sentence in tqdm(self.data):
            chunk_sentence = sentence["chunk_sentence"]
            head = [int(i) for i in sentence["head"]]
            dep = [int(i) for i in sentence["dep"]]
            self.create_dependency_data(chunk_sentence, head, dep)

            index += 1
            # DEBUG
            # if index == 100:
                # break
        # exit()

        self.len = len(self.dep_data)
        logger.info(f"dependency_data length: {self.len}")

    def __getitem__(self, index: int):
        return self.dep_data[index]

    def __len__(self) -> int:
        return self.len

    def create_dependency_data(
        self, chunk_sentence: list[str], head: list[int], dep: list[int]
    ):
        chunk_tokens, tokenized_sentence = self.tokenize_chunk(chunk_sentence)
        self.create_deps_matrix(dep, chunk_tokens, tokenized_sentence)

    def tokenize_chunk(self, chunk_sentence: list[str]):
        chunk_tokens = []
        sentence = "".join(chunk_sentence)
        for chunk in chunk_sentence:
            tokens = self.tokenizer.encode(chunk, add_special_tokens=False)
            chunk_tokens.append(tokens)
        tokenized_sentence = self.tokenizer.encode_plus(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=self.model_max_length,
            return_tensors="pt",
        )

        return chunk_tokens, tokenized_sentence

    def create_deps_matrix(self, dep, chunk_tokens, tokenized_sentence):
        token_clause_dict = self.create_token_clause_dict(chunk_tokens)
        token_len = len(token_clause_dict)

        before_special_token_num = 1
        for now_token_idx in range(token_len - 1):
            # 現在のトークンの後ろにあるトークンの数
            num_of_after_token = token_len - (now_token_idx + 1)
            now_token_head_idx = token_clause_dict[now_token_idx]
            now_token_dep_idx = dep[now_token_head_idx]
            is_last_token_in_clause = (
                now_token_head_idx != token_clause_dict[now_token_idx + 1]
            )

            for scope_token_idx in range(
                now_token_idx + 1, num_of_after_token + now_token_idx + 1
            ):
                scope_token_head_idx = token_clause_dict[scope_token_idx]

                # CLSトークンを考慮したインデックスになっている
                scope = torch.tensor(
                    [
                        now_token_idx + before_special_token_num,
                        scope_token_idx + before_special_token_num,
                    ],
                )

                is_first_token_in_clause = (
                    scope_token_head_idx != token_clause_dict[scope_token_idx - 1]
                )

                debug_dict = {
                    "now_token_idx": now_token_idx,
                    "scope_token_idx": scope_token_idx,
                    "scope": scope,
                    "now_token_head_idx": now_token_head_idx,
                    "scope_token_head_idx": scope_token_head_idx,
                    "now_token_dep_idx": now_token_dep_idx,
                    "token_clause_dict": token_clause_dict,
                    "is_last_token_in_clause": is_last_token_in_clause,
                    "is_first_token_in_clause": is_first_token_in_clause,
                }

                # 係元のtokenが文節内最後のtokenで，係先のtokenが文節内最初のtoken，
                # かつ係受け関係にある場合
                if (
                    now_token_dep_idx == scope_token_head_idx
                    and is_last_token_in_clause
                    and is_first_token_in_clause
                ):  # pos_outside
                    if scope[0] + 1 == scope[1]:
                        type_name = "pos_outside_neighbor"
                        self.pos_outside_neighbor_num += 1
                    else:
                        type_name = "pos_outside_distant"
                        self.pos_outside_distant_num += 1
                    self.pos_outside_num += 1

                    self.dep_data.append(
                        {
                            "label": torch.tensor(1),
                            "token": tokenized_sentence,
                            "scope": scope,
                            "type": type_name,
                            "debug_dict": debug_dict,
                        }
                    )
                # 係元のtokenが係先のtokenの隣の場合(文節内での係り受け)
                elif (  # pos_inside
                    now_token_head_idx == scope_token_head_idx
                    and now_token_idx + 1 == scope_token_idx
                ):
                    self.pos_inside_neighbor_num += 1
                    self.dep_data.append(
                        {
                            "label": torch.tensor(1),
                            "token": tokenized_sentence,
                            "scope": scope,
                            "type": "pos_inside_neighbor",
                            "debug_dict": debug_dict,
                        }
                    )
                elif now_token_idx + 1 == scope_token_idx:  # neg_neighbor
                    self.neg_neighbor_num += 1
                    self.dep_data.append(
                        {
                            "label": torch.tensor(0),
                            "token": tokenized_sentence,
                            "scope": scope,
                            "type": "neg_neighbor",
                            "debug_dict": debug_dict,
                        }
                    )
                else:  # neg_distant
                    self.neg_distant_num += 1
                    self.dep_data.append(
                        {
                            "label": torch.tensor(0),
                            "token": tokenized_sentence,
                            "scope": scope,
                            "type": "neg_distant",
                            "debug_dict": debug_dict,
                        }
                    )

    def create_token_clause_dict(self, chunk_tokens):
        scope_dict = {}
        now_scope = 0
        for i, chunk in enumerate(chunk_tokens):
            for _ in range(len(chunk)):
                scope_dict[now_scope] = i
                now_scope += 1
        return scope_dict

    def get_info(self):
        return {
            "pos_outside_num": self.pos_outside_num,
            "pos": {
                "pos_outside_neighbor_num": self.pos_outside_neighbor_num,
                "pos_outside_distant_num": self.pos_outside_distant_num,
            },
            "pos_inside_neighbor_num": self.pos_inside_neighbor_num,
            "neg_neighbor_num": self.neg_neighbor_num,
            "neg_distant_num": self.neg_distant_num,
        }
