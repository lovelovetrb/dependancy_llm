# Usage: knbc2cabocha.py [format]
#   convert KNB corpus to CaboCha format
import argparse
import glob
import re
import sys

from tqdm import tqdm


class KNP:
    def __init__(self, debug=False) -> None:
        self.debug = debug
        self.num_sentence = 0
        self.parsed_knp_data = []
        (
            self.header,
            self.chunk,
            self.chunk_surface,
            self.head,
            self.dep,
            self.body,
            self.t2c,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
            {},
        )

    def analyze_ktc_file(self, file_lines: list[str]):
        pat_d = re.compile(r"^\d+")

        for line in file_lines:
            line = line.strip()
            line = line.split(" ")

            # idに関する情報
            if line[0] == "#":
                self.header = line

            # 文末処理
            elif line[0] == "EOS":
                try:
                    self.head = [
                        pat_d.sub(lambda m: self.t2c[m.group()], h) for h in self.head
                    ]
                    for i in range(len(self.body)):
                        chunk_body = self.body[i].split("\n")
                        # chunk_bodyから空の要素を削除
                        chunk_body = list(filter(lambda x: x != "", chunk_body))
                        chunk_body = [c.split() for c in chunk_body]
                        self.chunk.append(chunk_body)
                        surface = ""
                        for index, c in enumerate(chunk_body):
                            surface += c[0]
                            if index == len(chunk_body) - 1:
                                self.chunk_surface.append(surface)
                    assert len(self.head) == len(self.dep) == len(self.chunk_surface)

                    ########### DEBUG ############
                    if self.debug:
                        print(f"header        : {self.header}       ")
                        print(f"chunk         : {self.chunk}        ")
                        print(f"chunk_surface : {self.chunk_surface}")
                        print(f"head          : {self.head}         ")
                        print(f"dep           : {self.dep}          ")
                    ##############################

                except KeyError:
                    raise ValueError("fail to convert: %s" % self.header)
                self.end_sentence()

            # *始まりは係り受けに関する情報
            elif line[0] == "*" or line[0] == "+":
                if line[0] == "*":
                    self.body.append("")
                    self.head.append("")
                    self.dep.append("")
                    assert len(self.head) == len(self.dep)
                self.head[-1] = line[1]
                # line[2]から英字を削除
                line[2] = re.sub(r"[a-zA-Z]", "", line[2])
                self.dep[-1] = line[2]
                self.t2c[str(len(self.t2c))] = str(len(self.head) - 1)

            # それ以外は形態素解析に関する情報
            else:
                if len(line) >= 2:  # KNP
                    if line[0] == line[2]:
                        line[2] = "*"
                    self.body[-1] += " ".join(line[0:3] + line[3:10:2] + ["*"]) + "\n"
                else:
                    self.body[-1] += "%s\t%s\n" % (
                        line[0],
                        ",".join(line[3:10:2] + [line[2], line[1], "*"]),
                    )

    def end_sentence(self):
        self.num_sentence += 1
        self.parsed_knp_data.append(
            {
                "id": self.header[1].replace("S-ID:", ""),
                "chunk_sentence": self.chunk_surface,
                "head": self.head,
                "dep": self.dep,
            }
        )
        (
            self.header,
            self.chunk,
            self.chunk_surface,
            self.head,
            self.dep,
            self.body,
            self.t2c,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
            {},
        )
        assert (
            len(self.header) == 0
            and len(self.chunk) == 0
            and len(self.chunk_surface) == 0
            and len(self.head) == 0
            and len(self.dep) == 0
            and len(self.body) == 0
        )

    def reset(self):
        self.num_sentence = 0

    def get_info(self):
        return self.num_sentence


def main():
    # init args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", type=str, default="./ktc_corpus_utf8/")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # file list
    files = glob.glob(args.dir_path + "*.ntc")
    progress = tqdm(files, desc="progress", disable=args.debug)
    can_use_sentence = 0
    knp = KNP(debug=args.debug)
    for file in progress:
        with open(file, "r") as f:
            data = f.readlines()
        knp.analyze_ktc_file(data)
        num_sentence = knp.get_info()
        can_use_sentence += num_sentence
        knp.reset()
        if args.debug:
            print(f"file: {file} num_sentence: {num_sentence}")
            knp.write_json_data("./test.json")
            break

    train_data, test_data = split_data_by_ratios(knp.parsed_knp_data, 2, [0.8, 0.2])
    print(f"train data length: {len(train_data)}")
    print(f"test data length: {len(test_data)}")

    json_train_data = convert_to_json(train_data)
    json_test_data = convert_to_json(test_data)

    write_json_data("./ktc_train_data.json", json_train_data)
    write_json_data("./ktc_test_data.json", json_test_data)
    print(f"can_use_sentence: {can_use_sentence}")


def split_data_by_ratios(data, num_splits, ratios):
    assert len(ratios) == num_splits, "分割個数と割合のリストの長さが一致しません。"
    assert sum(ratios) == 1, "割合の合計が1になっていません。"

    total_length = len(data)
    splits = []
    start = 0

    for ratio in ratios:
        end = start + int(ratio * total_length)
        splits.append(data[start:end])
        start = end

    return splits


def convert_to_json(dic_data: dict):
    import json

    # ensure_ascii: 非ASCII文字(日本語など)をエスケープするかどうか
    json_data = json.dumps(dic_data, indent=4, ensure_ascii=False)
    return json_data


def write_json_data(file_path, json_data):
    with open(file_path, "w", encoding="utf_8") as f:
        f.write(json_data)
    print(f"write: {file_path}")


if __name__ == "__main__":
    main()
