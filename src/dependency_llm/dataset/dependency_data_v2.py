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
        self.dep_type = []
        self.dep_label = []

        # DEBUG
        index = 0
        for sentence in tqdm(self.data):
            chunk_sentence = sentence["chunk_sentence"]
            head = [int(i) for i in sentence["head"]]
            dep = [int(i) for i in sentence["dep"]]
            (
                one_sentence_label,
                one_sentence_type,
                one_sentence_dep_data,
            ) = self.create_dependency_data(chunk_sentence, head, dep)

            if len(one_sentence_label) == 0:
                continue
            self.dep_data.append(one_sentence_dep_data)
            self.dep_type.append(one_sentence_type)
            self.dep_label.append(torch.tensor(one_sentence_label))
            index += 1
            # DEBUG
            if index == 100:
                break
        # exit()

        assert len(self.dep_data) == len(self.dep_label) == len(self.dep_type)
        self.len = len(self.dep_data)
        logger.info(f"dependency_data length: {self.len}")

    def __getitem__(self, index: int):
        return (self.dep_label[index], self.dep_type[index], self.dep_data[index])

    def __len__(self) -> int:
        return self.len

    def create_dependency_data(
        self, chunk_sentence: list[str], head: list[int], dep: list[int]
    ):
        chunk_tokens, tokenized_sentence = self.tokenize_chunk(chunk_sentence)
        (
            one_sentence_label,
            one_sentence_type,
            one_sentence_data,
        ) = self.create_deps_matrix(dep, chunk_tokens, tokenized_sentence)
        return one_sentence_label, one_sentence_type, one_sentence_data

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

        one_sentence_dep_data = []
        one_sentence_type = []
        one_sentence_label = []
        before_special_token_num = 1
        for now_token_idx in range(token_len - 1):
            # 現在のトークンの後ろにあるトークンの数
            num_of_after_token = token_len - (now_token_idx + 1)
            now_token_head_idx = token_clause_dict[now_token_idx]
            now_token_dep_idx = dep[now_token_head_idx]
            is_last_token_in_clause = (
                now_token_head_idx != token_clause_dict[now_token_idx + 1]
            )

            temp_one_dep_data = []
            temp_one_type = []
            temp_one_label = []
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
                    else:
                        type_name = "pos_outside_distant"
                    temp_one_label.append(1)
                    temp_one_type.append(type_name)
                    temp_one_dep_data.append(
                        {
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
                    temp_one_label.append(1)
                    temp_one_type.append("pos_inside_neighbor")
                    temp_one_dep_data.append(
                        {
                            "token": tokenized_sentence,
                            "scope": scope,
                            "type": "pos_inside_neighbor",
                            "debug_dict": debug_dict,
                        }
                    )
                elif now_token_idx + 1 == scope_token_idx:  # neg_neighbor
                    temp_one_label.append(0)
                    temp_one_type.append("neg_neighbor")
                    temp_one_dep_data.append(
                        {
                            "token": tokenized_sentence,
                            "scope": scope,
                            "type": "neg_neighbor",
                            "debug_dict": debug_dict,
                        }
                    )
                else:  # neg_distant
                    temp_one_label.append(0)
                    temp_one_type.append("neg_distant")
                    temp_one_dep_data.append(
                        {
                            "token": tokenized_sentence,
                            "scope": scope,
                            "type": "neg_distant",
                            "debug_dict": debug_dict,
                        }
                    )

            # DEBUG
            # print(temp_one_label)
            try:
                assert temp_one_label.count(1) == 1 and all(
                    x in [0, 1] for x in temp_one_label
                )
            except:
                if torch.distributed.get_rank() == 0:
                    print(f"temp_one_label: {temp_one_label} {temp_one_label.count(1)}")
                    print(temp_one_type)
                    print(temp_one_dep_data[0]["token"]["input_ids"])
                    print(temp_one_dep_data[0]["debug_dict"])
                    print("\n")

            assert (
                len(temp_one_dep_data)
                == len(temp_one_label)
                == len(temp_one_type)
                == num_of_after_token
            )

            # DEBUG
            # if torch.distributed.get_rank() == 0:
            # print(temp_one_label)

            one_sentence_dep_data.append(temp_one_dep_data)
            one_sentence_type.extend(temp_one_type)
            one_sentence_label.extend(temp_one_label)

        pred_data_num = 0
        for i in range(token_len - 1):
            pred_data_num += token_len - (i + 1)

        assert len(one_sentence_label) == len(one_sentence_type) == pred_data_num
        assert len(one_sentence_dep_data) == (token_len - 1)
        return one_sentence_label, one_sentence_type, one_sentence_dep_data

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
            "hard_ok_num": self.hard_ok_num,
            "ok_num": self.ok_num,
            "hard_ng_num": self.hard_ng_num,
            "ng_num": self.ng_num,
        }
