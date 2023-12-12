import argparse
import os

import torch
import yaml
from transformers import BertJapaneseTokenizer

import wandb
from src.dependency_llm.dependency_data import dependency_data
from src.dependency_llm.dependencyMatrixModel import DependencyMatrixModel
from src.dependency_llm.trainer import Trainer
from src.dependency_llm.util import init_config, init_gpu, logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # config.yamlの読み込み
    logger.info("Loading config.yaml...")
    with open("src/dependency_llm/config.yaml") as f:
        config = yaml.safe_load(f)

    if len(config["gpu"]["visible_gpu"]) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(i) for i in config["gpu"]["visible_gpu"]]
        )

    if not args.debug:
        init_gpu(args)
    init_config(config)

    tokenizer = BertJapaneseTokenizer.from_pretrained(config["basic"]["model_name"])
    args.model_max_length = tokenizer.max_model_input_sizes[
        config["basic"]["model_name"]
    ]

    logger.info("Loading model...")
    model = DependencyMatrixModel(
        config["basic"]["model_name"],
        add_layer_size=config["basic"]["add_layer_size"],
        max_length=args.model_max_length,
    )
    model.to("cuda")
    model.train()

    logger.info("Loading dataset...")
    dataset = dependency_data(
        data_path=config["basic"]["data_path"],
        model_max_length=args.model_max_length,
        tokenizer=tokenizer,
    )

    assert (
        config["dataset"]["train_ratio"]
        + config["dataset"]["valid_ratio"]
        + config["dataset"]["test_ratio"]
        == 1.0
    ), "train_ratio + valid_ratio + test_ratio must be 1.0"

    train_size = int(len(dataset) * config["dataset"]["train_ratio"])
    test_size = int(len(dataset) * config["dataset"]["test_ratio"])
    valid_size = int(len(dataset) * config["dataset"]["valid_ratio"])

    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, valid_size, test_size]
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        config=config,
        args=args,
    )
    trainer.train()

    logger.info("Finish training!")
    wandb.finish()


if __name__ == "__main__":
    main()
