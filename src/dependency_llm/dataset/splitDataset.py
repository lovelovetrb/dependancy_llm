import random

import torch
from dataset.dependency_data import dependency_data
from util import logger


class splitDataset:
    def __init__(self, config: dict, dataset: dependency_data):
        self.config = config
        self.dataset = dataset

        self.sanity_check(config)

        self.train_size = int(len(dataset) * config["dataset"]["train_ratio"])
        self.valid_size = int(len(dataset) * config["dataset"]["valid_ratio"])
        self.test_size = int(len(dataset) * config["dataset"]["test_ratio"])
        sum_size = self.train_size + self.valid_size + self.test_size

        if sum_size < len(dataset):
            self.train_size += len(dataset) - sum_size

    def split_dataset_for_training_and_evaluation(self) -> tuple:
        train_dataset = torch.utils.data.Subset(
            self.dataset, list(range(0, self.train_size))
        )
        valid_dataset = torch.utils.data.Subset(
            self.dataset,
            list(range(self.train_size, self.train_size + self.valid_size)),
        )
        test_dataset = torch.utils.data.Subset(
            self.dataset,
            list(
                range(
                    self.train_size + self.valid_size,
                    self.train_size + self.valid_size + self.test_size,
                )
            ),
        )
        return train_dataset, valid_dataset, test_dataset

    def adj_dataset(self, dataset: torch.utils.data.Subset) -> torch.utils.data.Subset:
        indeices = []
        pos_outside_num = 0
        pos_inside_neighbor_num = 0
        neg_neighbor_num = 0
        neg_distant_num = 0

        # dataをtypeごとに同じ数だけ取り出す
        for idx, data in enumerate(dataset):
            min_num = min(
                pos_outside_num,
                pos_inside_neighbor_num,
                neg_neighbor_num,
                neg_distant_num,
            )
            if (
                data["type"] == "pos_outside_neighbor"
                or data["type"] == "pos_outside_distant"
            ):
                if pos_outside_num <= min_num:
                    pos_outside_num += 1
                    indeices.append(idx)
            elif data["type"] == "pos_inside_neighbor":
                if pos_inside_neighbor_num <= min_num:
                    pos_inside_neighbor_num += 1
                    indeices.append(idx)
            elif data["type"] == "neg_neighbor":
                if neg_neighbor_num <= min_num:
                    neg_neighbor_num += 1
                    indeices.append(idx)
            elif data["type"] == "neg_distant":
                if neg_distant_num <= min_num:
                    neg_distant_num += 1
                    indeices.append(idx)
            else:
                continue
        logger.info("=== adj_train_dataset ===")
        logger.info(f"pos_outside_num: {pos_outside_num}")
        logger.info(f"pos_inside_neighbor_num: {pos_inside_neighbor_num}")
        logger.info(f"neg_neighbor_num: {neg_neighbor_num}")
        logger.info(f"neg_distant_num: {neg_distant_num}")
        logger.info("==========================")

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
        random.shuffle(indeices)

        return torch.utils.data.Subset(dataset, indeices)

    def sanity_check(self, config: dict):
        assert (
            config["dataset"]["train_ratio"]
            + config["dataset"]["valid_ratio"]
            + config["dataset"]["test_ratio"]
            == 1.0
        ), "train_ratio + valid_ratio + test_ratio must be 1.0"
