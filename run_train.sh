rye run torchrun --rdzv_endpoint localhost:29557  --nnodes=1 --node_rank=0 --nproc_per_node=2 src/dependency_llm/train.py --config_path "src/dependency_llm/config/setting.yaml"
