import random

import torch
from dataset.dependency_data import dependency_data
from util import logger


class splitDataset:
    def __init__(self, config: dict, dataset: dependency_data):
        self.config = config
        self.dataset = dataset

    def split_dataset_for_training_and_evaluation(self) -> tuple:
        train_indices = []
        valid_indices = []
        test_indices = []
        for idx, data in enumerate(self.dataset):
            if data["data_split_type"] == "train":
                train_indices.append(idx)
            elif data["data_split_type"] == "valid":
                valid_indices.append(idx)
            elif data["data_split_type"] == "test":
                test_indices.append(idx)
            else:
                raise ValueError("data_split_type is invalid")

        train_dataset = torch.utils.data.Subset(self.dataset, train_indices)
        valid_dataset = torch.utils.data.Subset(self.dataset, valid_indices)
        test_dataset = torch.utils.data.Subset(self.dataset, test_indices)

        return train_dataset, valid_dataset, test_dataset

    def adj_dataset(self, dataset: torch.utils.data.Subset) -> torch.utils.data.Subset:
        indeices = []
        pos_outside_neighbor_num = 0
        pos_outside_distant_num = 0
        pos_inside_neighbor_num = 0
        neg_neighbor_num = 0
        neg_distant_num = 0

        # dataをtypeごとに同じ数だけ取り出す
        for idx, data in enumerate(dataset):
            min_num = min(
                pos_outside_neighbor_num,
                pos_outside_distant_num,
                pos_inside_neighbor_num,
                neg_neighbor_num,
                neg_distant_num,
            )
            if data["type"] == "pos_outside_neighbor":
                if pos_outside_neighbor_num <= min_num:
                    pos_outside_neighbor_num += 1
                    indeices.append(idx)
            elif data["type"] == "pos_outside_distant":
                if pos_outside_distant_num <= min_num:
                    pos_outside_distant_num += 1
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
        logger.info(
            f"pos_outside_num: {pos_outside_neighbor_num + pos_outside_distant_num}"
        )
        logger.info(f"   - pos_outside_neighbor_num: {pos_outside_neighbor_num}")
        logger.info(f"   - pos_outside_distant_num: {pos_outside_distant_num}")
        logger.info(f"pos_inside_neighbor_num: {pos_inside_neighbor_num}")
        logger.info(f"neg_neighbor_num: {neg_neighbor_num}")
        logger.info(f"neg_distant_num: {neg_distant_num}")
        logger.info("==========================")

        assert (
            max(
                pos_outside_neighbor_num,
                pos_outside_distant_num,
                pos_inside_neighbor_num,
                neg_neighbor_num,
                neg_distant_num,
            )
            - min(
                pos_outside_neighbor_num,
                pos_outside_distant_num,
                pos_inside_neighbor_num,
                neg_neighbor_num,
                neg_distant_num,
            )
            <= 1
        )
        random.shuffle(indeices)
        return torch.utils.data.Subset(dataset, indeices)
