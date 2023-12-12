import os

import torch
import torch.distributed as dist
from tqdm import tqdm

import wandb
from src.dependency_llm.dependency_data import dependency_data
from src.dependency_llm.dependencyMatrixModel import DependencyMatrixModel
from src.dependency_llm.util import logger


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

        self.model = model
        wandb.watch(model)

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.loss_fn = self.loss_fn.to(self.args.device)
        self.scaler = torch.cuda.amp.GradScaler()

        self.last_loss = float("inf")
        self.best_loss = float("inf")
        self.last_accuracy = float("-inf")
        self.best_accuracy = float("-inf")
        self.test_accuracy = float("-inf")
        self.now_epoch = 1
        self.now_iter = 1

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["train"]["lr"],
            weight_decay=self.config["train"]["weight_decay"],
        )

        train_sampler = torch.utils.data.DistributedSampler(
            self.train_dataset,
            num_replicas=self.args.world_size,
            rank=self.args.local_rank,
        )

        valid_sampler = torch.utils.data.DistributedSampler(
            self.valid_dataset,
            num_replicas=self.args.world_size,
            rank=self.args.local_rank,
        )

        test_sampler = torch.utils.data.DistributedSampler(
            self.test_dataset,
            num_replicas=self.args.world_size,
            rank=self.args.local_rank,
        )

        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config["train"]["batch_size"],
            sampler=train_sampler,
            num_workers=os.cpu_count(),
            pin_memory=True,
            drop_last=True,
        )

        self.valid_dataloader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.config["train"]["valid_batch_size"],
            sampler=valid_sampler,
            num_workers=os.cpu_count(),
            pin_memory=True,
            drop_last=True,
        )

        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config["train"]["test_batch_size"],
            sampler=test_sampler,
            num_workers=os.cpu_count(),
            pin_memory=True,
            drop_last=True,
        )

        wandb.log(
            {
                "train_dataset_size": len(self.train_dataset),
                "valid_dataset_size": len(self.valid_dataset),
                "test_dataset_size": len(self.test_dataset),
            }
        )
        # pytorchの表示制限を解除
        torch.set_printoptions(threshold=100000)
        torch.backends.cudnn.benchmark = True

    def train(self) -> None:
        logger.info("Start training...")
        for _ in range(self.config["train"]["epoch"]):
            dist.barrier()
            logger.info(f"Epoch: {self.now_epoch}")

            train_iter_bar = tqdm(
                self.train_dataloader,
                disable=not self.args.is_master,
            )
            ######## Training ########
            for batch_label, batch_data in train_iter_bar:
                train_iter_bar.set_postfix(loss=self.last_loss)
                self.step(batch_label=batch_label, batch_data=batch_data)
                wandb.log(
                    {
                        "epoch": self.now_epoch,
                        "iter": self.now_iter,
                        "loss": self.last_loss,
                    }
                )

                if self.last_loss < self.best_loss:
                    self.best_loss = self.last_loss
                    save_path = self.config["train"]["save_path"] + "best_loss.pth"
                    self.save_model(save_path)
                self.now_iter += 1

            ######## Validation ########
            valid_iter_bar = tqdm(
                self.valid_dataloader,
                disable=not self.args.is_master,
            )
            for batch_label, batch_data in valid_iter_bar:
                valid_iter_bar.set_postfix(accuracy=self.last_accuracy)
                self.valid(batch_label=batch_label, batch_data=batch_data)
                if self.last_accuracy > self.best_accuracy:
                    self.best_accuracy = self.last_accuracy
                wandb.log(
                    {
                        f"epoch {self.now_epoch} - accuracy": self.last_accuracy,
                    }
                )
            self.end_epoch()

        ######## Test ########
        self.test()

    def step(self, batch_label, batch_data) -> None:
        self.optimizer.zero_grad()
        batch_label = batch_label.to(self.args.device)  # [batch_size, seq_len, seq_len]

        with torch.cuda.amp.autocast():
            output = self.model(
                # [batch_size, 1, seq_len] -> [batch_size, seq_len]
                input_ids=batch_data.input_ids.view(
                    self.config["train"]["batch_size"], -1
                ).to(self.args.device),
                attention_mask=batch_data.attention_mask.view(
                    self.config["train"]["batch_size"], -1
                ).to(self.args.device),
            )

            assert batch_label.shape == output.shape
            loss = self.loss_fn(output, batch_label)

        self.scaler.scale(loss).backward()

        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.last_loss = loss.item()

    def valid(self, batch_label, batch_data) -> None:
        batch_label = batch_label.to(self.args.device)
        with torch.no_grad():
            output = self.model(
                input_ids=batch_data.input_ids.view(
                    self.config["train"]["valid_batch_size"], -1
                ).to(self.args.device),
                attention_mask=batch_data.attention_mask.view(
                    self.config["train"]["valid_batch_size"], -1
                ).to(self.args.device),
            )

            assert batch_label.shape == output.shape

        # outputのそれぞれの要素がマイナスであれば0に、プラスであれば1に変換
        output = torch.where(
            output < 0, torch.zeros_like(output), torch.ones_like(output)
        )
        # 正解率を計算
        mask = batch_label == 1
        accuracy = (output[mask] == batch_label[mask]).sum().item() / output[
            print(mask)
        ].numel()
        # 小数点以下第3位まで表示
        self.last_accuracy = round(accuracy, 4)

    def test(self) -> None:
        for batch_label, batch_data in self.test_dataloader:
            batch_label = batch_label.to(self.args.device)
            with torch.no_grad():
                output = self.model(
                    input_ids=batch_data.input_ids.view(
                        self.config["train"]["test_batch_size"], -1
                    ).to(self.args.device),
                    attention_mask=batch_data.attention_mask.view(
                        self.config["train"]["test_batch_size"], -1
                    ).to(self.args.device),
                )

                assert batch_label.shape == output.shape

            # outputのそれぞれの要素がマイナスであれば0に、プラスであれば1に変換
            output = torch.where(
                output < 0, torch.zeros_like(output), torch.ones_like(output)
            )

            # 正解率を計算(ラベルが1のときのみ計算)
            mask = batch_label == 1
            accuracy = (output[mask] == batch_label[mask]).sum().item() / output[
                mask
            ].numel()

            # 小数点以下第3位まで表示
            self.test_accuracy = round(accuracy, 4)

            wandb.log(
                {
                    "test_accuracy": self.last_accuracy,
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
