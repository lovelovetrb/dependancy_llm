import argparse
import os

import torch
import yaml
from data.dependency_data import dependency_data
from data.splitDataset import splitDataset
from model.baseModel import baseModel
from model.dependencyMatrixModel import DependencyMatrixModel
from trainer import Trainer
from transformers import BertJapaneseTokenizer
from util import init_config, init_gpu, logger

import wandb


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # config.yamlの読み込み
    logger.info("Loading config.yaml...")
    with open("src/dependency_llm/config.yaml") as f:
        config = yaml.safe_load(f)

    if len(config["gpu"]["visible_gpu"]) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(i) for i in config["gpu"]["visible_gpu"]]
        )

    init_gpu(args)
    init_config(config)

    tokenizer = BertJapaneseTokenizer.from_pretrained(config["model"]["model_name"])
    args.model_max_length = tokenizer.max_model_input_sizes[
        config["model"]["model_name"]
    ]

    logger.info("Loading dataset...")
    dataset = dependency_data(
        data_path=config["basic"]["data_path"],
        model_max_length=args.model_max_length,
        tokenizer=tokenizer,
    )

    split_dataset = splitDataset(config=config, dataset=dataset)

    (
        train_dataset,
        valid_dataset,
        test_dataset,
    ) = split_dataset.split_dataset_for_training_and_evaluation()

    train_dataset = split_dataset.adj_dataset(train_dataset)
    print(len(train_dataset))
    pos_outside_num = 0
    pos_inside_neighbor_num = 0
    neg_neighbor_num = 0
    neg_distant_num = 0
    for data in train_dataset:
        if (
            data["type"] == "pos_outside_neighbor"
            or data["type"] == "pos_outside_distant"
        ):
            pos_outside_num += 1
        elif data["type"] == "pos_inside_neighbor":
            pos_inside_neighbor_num += 1
        elif data["type"] == "neg_neighbor":
            neg_neighbor_num += 1
        elif data["type"] == "neg_distant":
            neg_distant_num += 1
        else:
            raise ValueError
    print(f"pos_outside_num: {pos_outside_num}")
    print(f"pos_inside_neighbor_num: {pos_inside_neighbor_num}")
    print(f"neg_neighbor_num: {neg_neighbor_num}")
    print(f"neg_distant_num: {neg_distant_num}")
    assert (
        max(
            pos_outside_num,
            pos_inside_neighbor_num,
            neg_neighbor_num,
            neg_distant_num,
        )
        - min(
            pos_outside_num,
            pos_inside_neighbor_num,
            neg_neighbor_num,
            neg_distant_num,
        )
        <= 1
    )
    exit()
    if config["basic"]["mode"] == "train":
        logger.info("Loading model...")
        base_model = DependencyMatrixModel(
            model_name=config["model"]["model_name"],
            scope_layer_size=config["model"]["scope_layer_size"],
            max_length=args.model_max_length,
            dropout=config["model"]["dropout"],
            hidden_size=config["model"]["hidden_size"],
            freeze_bert=config["model"]["bert_freeze"],
        )
        base_model.to("cuda")
        base_model.train()

        trainer = Trainer(
            model=base_model,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            config=config,
            args=args,
        )

        trainer.train()
        logger.info("Finish training!")
    elif config["basic"]["mode"] == "test":
        logger.info("Test Mode...")
        logger.info("Loading model...")

        if config["test"]["test_base_model"]:
            test_model = baseModel(
                model_name=config["model"]["model_name"],
                scope_layer_size=config["model"]["scope_layer_size"],
                max_length=args.model_max_length,
                num_labels=config["model"]["num_labels"],
                dropout=config["model"]["dropout"],
                hidden_size=config["model"]["hidden_size"],
                freeze_bert=config["model"]["bert_freeze"],
            )
            if config["test"]["knockout"]:
                raise ValueError("knockout is not available in baseModel")
        else:
            test_model = torch.load(config["train"]["save_path"] + "best_loss.pth")

        test_model.eval()
        test_model.to("cuda")

        if config["test"]["knockout"]:
            test_model.knockout(config["test"]["knockout_layer_head"])
        else:
            logger.info("No knockout")

        trainer = Trainer(
            model=test_model,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
            config=config,
            args=args,
        )
        trainer.test()
        logger.info("Finish testing!")
    else:
        logger.info("Invalid mode")
        raise ValueError

    wandb.finish()


def get_dataset_config(dataset: dependency_data):
    correct_num = 0
    bad_num = 0
    for data in dataset:
        if data["label"] == 0:
            bad_num += 1
        elif data["label"] == 1:
            correct_num += 1
        else:
            raise ValueError
    return correct_num, bad_num


if __name__ == "__main__":
    main()
