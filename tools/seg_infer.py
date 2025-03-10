import argparse
import yaml
from holitracer.seg.engine import InferImageEngine
import setproctitle
import torch.distributed as dist
import torch

import os

setproctitle.setproctitle("python seg_infer.py")


def parse_args():
    parser = argparse.ArgumentParser(description="Inference Engine")
    parser.add_argument("--config", type=str, default='./seg/config/whubuilding_infer.yaml', help="Path to the config file")

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers (not used without DataLoader).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (not used without DataLoader).",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Local rank for distributed processing.",
    )

    cli_args = parser.parse_args()

    # If a config file is provided, load it and override default arguments
    if cli_args.config:
        with open(cli_args.config, "r") as f:
            config_args = yaml.safe_load(f)
        # Update CLI arguments with config file arguments
        for key, value in config_args.items():
            setattr(cli_args, key, value)

    args = cli_args

    # Read local_rank, rank, and world_size from environment variables
    args.local_rank = int(os.getenv("LOCAL_RANK", args.local_rank))
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
    else:
        torch.cuda.set_device(args.local_rank)

    engine = InferImageEngine(args)
    engine.predict()

    if args.distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
