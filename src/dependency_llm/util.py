import logging

import torch
import torch.distributed as dist
from transformers import set_seed

import wandb

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - PID: %(process)d -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def init_config(config: dict):
    logger.info("Initializing wandb...")
    logger.info(f"seed: {config['basic']['seed']}")
    set_seed(config["basic"]["seed"])
    wandb.init(
        project="dependency_llm",
        config=config,
    )


def init_gpu(args):
    logger.info("Initializing GPUs...")
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())
    local_rank = dist.get_rank()
    args.local_rank = local_rank
    args.world_size = dist.get_world_size()
    args.is_master = local_rank == 0
    args.device = torch.device("cuda")
