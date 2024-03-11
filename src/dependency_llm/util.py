import logging

import torch
import torch.distributed as dist
from transformers import set_seed, BertJapaneseTokenizer

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

def init_tokenizer(model_name:str):
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
    return tokenizer

def get_model_max_length(tokenizer, model_name:str):
    return tokenizer.max_model_input_sizes[model_name]

def detect_type(scope, now_token_idx, scope_token_idx, token_clause_dict, dep):
    now_token_head_idx = token_clause_dict[now_token_idx]
    now_token_dep_idx = dep[now_token_head_idx]
    scope_token_head_idx = token_clause_dict[scope_token_idx]
    
    is_first_token_in_clause = (
        scope_token_head_idx != token_clause_dict[scope_token_idx - 1]
    )
    is_last_token_in_clause = (
        now_token_head_idx != token_clause_dict[now_token_idx + 1]
    )

    if (
        now_token_dep_idx == scope_token_head_idx
        and is_last_token_in_clause
        and is_first_token_in_clause
    ): 
        if scope[0]+1 == scope[1]:
            return "pos_outside_neighbor"
        else:
            return "pos_outside_distant"
    elif now_token_head_idx == scope_token_head_idx and now_token_idx + 1 == scope_token_idx:
        return "pos_inside_neighbor"
    elif now_token_idx + 1 == scope_token_idx:
        return "neg_neighbor"
    else:
        return "neg_distant"

def create_token_clause_dict(chunk_tokens):
    scope_dict = {}
    now_scope = 0
    for i, chunk in enumerate(chunk_tokens):
        for _ in range(len(chunk)):
            scope_dict[now_scope] = i
            now_scope += 1
    return scope_dict

def tokenize_chunk(tokenizer, chunk_sentence: list[str]):
    chunk_tokens = []
    for chunk in chunk_sentence:
        tokens = tokenizer.encode(chunk, add_special_tokens=False)
        chunk_tokens.append(tokens)

    return chunk_tokens