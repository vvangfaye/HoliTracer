import os
import yaml
import argparse
import setproctitle
import torch.distributed as dist
import torch
from holitracer.vector.engine import TrainEngine

setproctitle.setproctitle("python vector_train.py")


def parse_args():
    parser = argparse.ArgumentParser(description="Segmentation Engine")
    parser.add_argument(
        "--config",
        type=str,
        default="./vector/config/glhwater_train.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="Local rank for distributed training"
    )

    cli_args = parser.parse_args()
    with open(cli_args.config, "r") as f:
        config_args = yaml.safe_load(f)
    args = argparse.Namespace(**config_args)

    # 从环境变量中读取 local_rank、rank 和 world_size
    args.local_rank = int(os.getenv("LOCAL_RANK", 0))
    args.rank = int(os.getenv("RANK", 0))
    args.world_size = int(os.getenv("WORLD_SIZE", 1))
    args.distributed = args.world_size > 1

    print(
        f"Rank: {args.rank}, World Size: {args.world_size}, Local Rank: {args.local_rank}"
    )
    return args


def main():
    args = parse_args()
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        
    train_engine = TrainEngine(args)

    train_engine.train()

    if args.distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()