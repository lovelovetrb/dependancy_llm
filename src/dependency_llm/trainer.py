import os

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from data.dependency_data import dependency_data
from model.dependencyMatrixModel import DependencyMatrixModel
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score)
from tqdm import tqdm
from util import logger

import wandb


class Trainer:
    def __init__(
        self,
        model: DependencyMatrixModel,
        train_dataset: dependency_data,
        valid_dataset: dependency_data,
        test_dataset: dependency_data,
        config: dict,
        args: dict,
    ) -> None:
        logger.info("Initializing trainer...")
        self.config = config
        self.args = args

        if os.path.exists(self.config["train"]["save_path"]) is False:
            os.makedirs(self.config["train"]["save_path"])

        self.model = model
        wandb.watch(model)

        self.last_loss = float("inf")
        self.best_loss = float("inf")
        self.last_accuracy = float("-inf")
        self.best_accuracy = float("-inf")
        self.test_accuracy = float("-inf")
        self.now_epoch = 1
        self.now_iter = 1

        if self.config["basic"]["mode"] == "train":
            if self.config["train"]["fp16"]:
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.config["train"]["lr"],
                    weight_decay=self.config["train"]["weight_decay"],
                )
                self.scaler = torch.cuda.amp.GradScaler()
            else:
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=self.config["train"]["lr"],
                    weight_decay=self.config["train"]["weight_decay"],
                )
            self.loss_fn = torch.nn.CrossEntropyLoss()
            self.loss_fn = self.loss_fn.to(self.args.device)

        train_sampler = torch.utils.data.DistributedSampler(
            train_dataset,
            num_replicas=self.args.world_size,
            rank=self.args.local_rank,
            shuffle=True,
        )

        valid_sampler = torch.utils.data.DistributedSampler(
            valid_dataset,
            num_replicas=self.args.world_size,
            rank=self.args.local_rank,
            shuffle=True,
        )

        test_sampler = torch.utils.data.DistributedSampler(
            test_dataset,
            num_replicas=self.args.world_size,
            rank=self.args.local_rank,
            shuffle=True,
        )

        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config["train"]["batch_size"],
            sampler=train_sampler,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )

        self.valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.config["train"]["valid_batch_size"],
            sampler=valid_sampler,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )

        self.test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config["train"]["test_batch_size"],
            sampler=test_sampler,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )
        if self.args.is_master:
            logger.info(f"Train dataloader length: {len(self.train_dataloader)}")
            logger.info(f"Valid dataloader length: {len(self.valid_dataloader)}")
            logger.info(f"Test dataloader length: {len(self.test_dataloader)}")
        wandb.log(
            {
                "train_dataset_length": len(self.train_dataloader),
                "valid_dataset_length": len(self.valid_dataloader),
                "test_dataset_length": len(self.test_dataloader),
            }
        )
        dist.barrier()

    def train(self) -> None:
        logger.info("Start training...")
        for _ in range(self.config["train"]["epoch"]):
            ######## Training ########
            if self.args.is_master:
                logger.info("######## Training ########")
                logger.info(f"Epoch: {self.now_epoch}")
            train_iter_bar = tqdm(
                self.train_dataloader,
                disable=not self.args.is_master,
            )

            for batch_label, _, batch_data in train_iter_bar:
                if self.args.is_master:
                    train_iter_bar.set_postfix(loss=self.last_loss)
                self.train_step(batch_label=batch_label, batch_data=batch_data)
                wandb.log(
                    {
                        "epoch": self.now_epoch,
                        "iter": self.now_iter,
                        "loss": self.last_loss,
                    }
                )

                if self.last_loss < self.best_loss:
                    self.best_loss = self.last_loss
                    if self.args.is_master:
                        save_path = self.config["train"]["save_path"] + "best_loss.pth"
                        self.save_model(save_path)
                self.now_iter += 1

            ######## Validation ########
            if self.args.is_master:
                logger.info("######## Validation ########")
            self.valid()
            self.end_epoch()

    def train_step(self, batch_label, batch_data) -> None:
        self.optimizer.zero_grad()
        batch_label = batch_label.to(self.args.device)

        with torch.cuda.amp.autocast():
            output = self.model(batch_data)
            assert output.shape == batch_label.shape
            loss = self.loss_fn(output, batch_label)

        if self.config["train"]["fp16"]:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        self.last_loss = loss.item()

    def valid(self) -> None:
        valid_iter_bar = tqdm(
            self.valid_dataloader,
            disable=not self.args.is_master,
        )
        correct_label = []
        pred_label = []
        for batch_label, _, batch_data in valid_iter_bar:
            with torch.no_grad():
                output = self.model(batch_data)
                try:
                    correct_label += batch_label.squeeze().tolist()
                    pred_label += output.cpu().squeeze().tolist()
                except:
                    print(batch_label)
                    print(output)
                    print(batch_data)

        assert len(pred_label) == len(correct_label)

        accuracy = accuracy_score(
            correct_label,
            pred_label,
        )
        precision = precision_score(
            correct_label,
            pred_label,
        )
        recall = recall_score(
            correct_label,
            pred_label,
        )
        f1 = f1_score(
            correct_label,
            pred_label,
        )

        self.last_accuracy = accuracy

        if self.args.is_master:
            logger.info(f"Epoch {self.now_epoch} - Accuracy: {accuracy}")
            logger.info(f"Epoch {self.now_epoch} - Precision: {precision}")
            logger.info(f"Epoch {self.now_epoch} - Recall: {recall}")
            logger.info(f"Epoch {self.now_epoch} - F1 Score: {f1}")
        wandb.log(
            {
                "epoch": self.now_epoch,
                "valid_accuracy": self.last_accuracy,
                "valid_precision": precision,
                "valid_recall": recall,
                "valid_f1_score": f1,
            }
        )

    def test(self) -> None:
        iter_bar = tqdm(
            self.test_dataloader,
            disable=not self.args.is_master,
        )
        correct_label_dic = {"all": []}
        pred_label_dic = {"all": []}
        from transformers import BertJapaneseTokenizer

        tokenizer = BertJapaneseTokenizer.from_pretrained(
            self.config["model"]["model_name"]
        )
        for batch_label, batch_type, batch_data in iter_bar:
            with torch.no_grad():
                output = self.model(batch_data)
            batch_label_list = batch_label.squeeze().tolist()
            output_list = output.cpu().squeeze().tolist()

            # DEBUG
            # if self.args.is_master:
            #     tokens = tokenizer.convert_ids_to_tokens(
            #         batch_data[0][0]["token"]["input_ids"].squeeze().squeeze(),
            #         skip_special_tokens=True,
            #     )
            #     print(tokens)
            #     for i in range(len(batch_data)):
            #         start_num = len(tokens) * i - (i * (i + 1) // 2)
            #         print("  ", end="")
            #         for _ in range(i):
            #             print("    ", end="")
            #         for j in range(len(batch_data[i])):
            #             num = start_num + j
            #             print(batch_label_list[num], end=" , ")
            #         print()
            #         print("  ", end="")
            #         for _ in range(i):
            #             print("    ", end="")
            #         for j in range(len(batch_data[i])):
            #             num = start_num + j
            #             print(int(output_list[num]), end=" , ")
            #         print()

            #     print()

            if isinstance(batch_label_list, float) or isinstance(output_list, int):
                # print(batch_label_list)
                # print(output_list)
                continue
            for label, type_name, pred in zip(
                batch_label_list,
                batch_type,
                output_list,
            ):
                correct_label_dic["all"].append(label)
                pred_label_dic["all"].append(pred)
                type_name = type_name[0]
                if type_name not in correct_label_dic.keys():
                    correct_label_dic[type_name] = [label]
                    pred_label_dic[type_name] = [pred]
                else:
                    correct_label_dic[type_name].append(label)
                    pred_label_dic[type_name].append(pred)

        for pred_label, correct_label, type_name in zip(
            pred_label_dic.values(),
            correct_label_dic.values(),
            correct_label_dic.keys(),
        ):
            assert len(pred_label) == len(correct_label)

            accuracy = accuracy_score(
                correct_label,
                pred_label,
            )
            precision = precision_score(
                correct_label,
                pred_label,
                zero_division=0,
            )
            recall = recall_score(
                correct_label,
                pred_label,
                zero_division=0,
            )
            f1 = f1_score(
                correct_label,
                pred_label,
                zero_division=0,
            )

            if self.args.is_master:
                logger.info(f"========== TEST SUMMARY - {type_name} ==========")
                logger.info(f"Test Accuracy ({type_name}) : {accuracy}")
                logger.info(f"Test Precision({type_name}) : {precision}")
                logger.info(f"Test Recall   ({type_name}) : {recall}")
                logger.info(f"Test F1 Score ({type_name}) : {f1}")
                if not self.config["test"]["test_base_model"]:
                    correct_label.append(0.0)
                    correct_label.append(1.0)
                    pred_label.append(0)
                    pred_label.append(1)

                    ConfusionMatrixDisplay.from_predictions(
                        correct_label,
                        pred_label,
                        display_labels=["bad", "correct"],
                        cmap=plt.cm.Blues,
                    )
                    plt.title(f"Confusion Matrix  - Accuracy: {accuracy}")
                    DIR_NAME = f"src/dependency_llm/fig/{self.config['basic']['seed']}/"
                    if os.path.exists(DIR_NAME) is False:
                        os.makedirs(DIR_NAME)
                    plt.savefig(
                        DIR_NAME + f"confusion_matrix_{type_name}.png",
                    )

            wandb.log(
                {
                    "type_name": type_name,
                    "test_accuracy": accuracy,
                    "test_precision": precision,
                    "test_recall": recall,
                    "test_f1_score": f1,
                }
            )

    def end_epoch(self) -> None:
        dist.barrier()
        if self.args.is_master:
            self.save_model(
                self.config["train"]["save_path"] + f"epoch_{self.now_epoch}_model.pth"
            )

        self.now_epoch += 1
        self.now_iter = 1

    def save_model(self, path) -> None:
        torch.save(self.model, path)
