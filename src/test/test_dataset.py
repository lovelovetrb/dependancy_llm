import torch
from instanse import TestInstance
from torch import tensor


def test_deps_data():
    test_instance = TestInstance()

    deps_data = test_instance.deps_data
    chunk_sentence = deps_data[0]["chunk_sentence"]
    senetence = "".join(chunk_sentence)
    tokens = test_instance.tokenizer.encode(senetence, add_special_tokens=False)

    assert len(test_instance.dataset) == 1

    test_dep_label, test_dep_type, test_dep_data = test_instance.dataset[0]

    assert len(test_dep_label) == len(test_dep_type) == 171

    data_num = 0
    for idx, label in enumerate(test_dep_data):
        for i, la in enumerate(label):
            data_num += 1
            if data_num - 1 in [
                0,
                18,
                37,
                51,
                66,
                80,
                102,
                105,
                116,
                132,
                135,
                143,
                153,
                156,
                161,
                165,
                168,
                170,
            ]:
                print(f"====== {data_num} ======")
                print(f"label: {test_dep_label[data_num - 1]}")
                print(test_dep_type[data_num - 1])
                assert test_dep_label[data_num - 1] == tensor(1)
                print(
                    test_instance.tokenizer.convert_ids_to_tokens(
                        tokens, skip_special_tokens=True
                    )
                )
                print(la["debug_dict"])
                print()
            else:
                assert test_dep_label[data_num - 1] == tensor(0)
    exit()

    pos_data_num = 0
    # for dataset_idx, label in test_subject.items():
    #     test_data = test_instance.dataset[dataset_idx]
    #
    #     print(f"====== {dataset_idx + 1} ======")
    #     print(f"type: {test_data['type']}")
    #     print(
    #         test_instance.tokenizer.convert_ids_to_tokens(
    #             test_data["token"]["input_ids"][0], skip_special_tokens=True
    #         )
    #     )
    #     print(f"label: {test_data['label']}")
    #     print(test_data["debug_dict"])
    #     print()
    #
    #     assert test_data["label"] == tensor(label)
    #     if test_data["label"] == tensor(1):
    #         assert torch.equal(test_data["scope"], pos_scope[pos_data_num])
    #         pos_data_num += 1
    #     assert test_data["token"]["input_ids"].shape == torch.Size(
    #         [1, test_instance.model_max_length]
    #     )
    #     assert test_data["token"]["attention_mask"].shape == torch.Size(
    #         [1, test_instance.model_max_length]
    #     )


if __name__ == "__main__":
    test_deps_data()
