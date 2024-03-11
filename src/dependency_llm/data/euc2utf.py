import glob
import os

from tqdm import tqdm


def main():
    DIR_PATH = "./knp/"
    OUT_DIR_PATH = "./ktc_corpus/"

    remove_files(OUT_DIR_PATH)
    convert_data(DIR_PATH, OUT_DIR_PATH)
    print("done")


def remove_files(dir_path):
    print(f"remove files in {dir_path}")
    files = glob.glob(dir_path + "*.ntc")
    for file in files:
        os.remove(file)


def convert_data(dir_path, out_dir_path):
    # load files
    # 拡張子がntcのファイルを取得
    files = glob.glob(dir_path + "*.ntc")

    # load deta
    index = 0
    for file in tqdm(files):
        file_name = os.path.basename(file)
        with open(file, "r", encoding="euc_jp") as f:
            data = f.readlines()
        for line in data:
            with open(out_dir_path + file_name, "a", encoding="utf_8") as f:
                f.write(line)
        index += 1


if __name__ == "__main__":
    main()
