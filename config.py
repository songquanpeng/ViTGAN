import argparse
import json
import os
import random
import shutil
import sys

import numpy as np
import torch
from munch import Munch
from torch.backends import cudnn

from utils.file import prepare_dirs
from utils.file import save_json
from utils.misc import get_datetime, str2bool, get_commit_hash


def setup_cfg(args):
    cudnn.benchmark = args.cudnn_benchmark
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    if args.mode == 'train' and torch.cuda.device_count() > 1:
        print(f"We will train on {torch.cuda.device_count()} GPUs.")
        args.multi_gpu = True
    else:
        args.multi_gpu = False

    if args.exp_id is None:
        args.exp_id = get_datetime()
        # Tip: you can construct the exp_id automatically here by use the args.

    if args.debug:
        print("Warning: running in debug mode, some settings will be override.")
        args.exp_id = "debug"
        args.sample_every = 10
        args.eval_every = 20
        args.save_every = 20
        args.end_iter = args.start_iter + 60
    if os.name == 'nt' and args.num_workers != 0:
        print("Warning: reset num_workers = 0, because running on a Windows system.")
        args.num_workers = 0

    args.log_dir = os.path.join(args.exp_dir, args.exp_id, "logs")
    args.sample_dir = os.path.join(args.exp_dir, args.exp_id, "samples")
    args.model_dir = os.path.join(args.exp_dir, args.exp_id, "models")
    args.eval_dir = os.path.join(args.exp_dir, args.exp_id, "eval")
    prepare_dirs([args.log_dir, args.sample_dir, args.model_dir, args.eval_dir])
    args.record_file = os.path.join(args.exp_dir, args.exp_id, "records.txt")
    args.loss_file = os.path.join(args.exp_dir, args.exp_id, "losses.csv")

    if os.path.exists(f'./scripts/{args.exp_id}.sh'):
        shutil.copyfile(f'./scripts/{args.exp_id}.sh', os.path.join(args.exp_dir, args.exp_id, f'{args.exp_id}.sh'))


def validate_cfg(args):
    assert args.eval_every % args.save_every == 0


def load_cfg():
    # There are two ways to load config, use a json file or command line arguments.
    if len(sys.argv) >= 2 and sys.argv[1].endswith('.json'):
        with open(sys.argv[1], 'r') as f:
            cfg = json.load(f)
            cfg = Munch(cfg)
            if len(sys.argv) >= 3:
                cfg.exp_id = sys.argv[2]
            else:
                print("Warning: using existing experiment dir.")
            if not cfg.about:
                cfg.about = f"Copied from: {sys.argv[1]}"
    else:
        cfg = parse_args()
        cfg = Munch(cfg.__dict__)
        if not cfg.hash:
            cfg.hash = get_commit_hash()
    current_hash = get_commit_hash()
    if current_hash != cfg.hash:
        print(f"Warning: unmatched git commit hash: `{current_hash}` & `{cfg.hash}`.")
    return cfg


def save_cfg(cfg):
    exp_path = os.path.join(cfg.exp_dir, cfg.exp_id)
    os.makedirs(exp_path, exist_ok=True)
    filename = cfg.mode
    if cfg.mode == 'train' and cfg.start_iter != 0:
        filename = f"resume_{cfg.start_iter}"
    save_json(exp_path, cfg, filename)


def print_cfg(cfg):
    print(json.dumps(cfg, indent=4))


def parse_args():
    parser = argparse.ArgumentParser()

    # About this experiment.
    parser.add_argument('--about', type=str, default="")
    parser.add_argument('--hash', type=str, required=False, help="Git commit hash for this experiment.")
    parser.add_argument('--exp_id', type=str, help='Folder name and id for this experiment.')
    parser.add_argument('--exp_dir', type=str, default='expr')

    # Meta arguments.
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'sample'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # Model related arguments.
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=8)
    parser.add_argument('--image_dim', type=int, default=3)
    parser.add_argument('--z_dim', type=int, default=64)
    parser.add_argument('--pool', type=str, default='cls', choices=['cls', 'mean'])
    parser.add_argument('--g_dim', type=int, default=384)
    parser.add_argument('--g_blocks', type=int, default=6)
    parser.add_argument('--g_attention_head_num', type=int, default=8)
    parser.add_argument('--g_attention_head_dim', type=int, default=64)
    parser.add_argument('--g_transformer_dropout', type=float, default=0.1)

    # Dataset related arguments.
    parser.add_argument('--dataset', type=str, default='CelebA')
    parser.add_argument('--dataset_path', type=str, required=True)

    # Training related arguments
    parser.add_argument('--parameter_init', type=str, default='he', choices=['he', 'default'])
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--end_iter', type=int, default=100000)
    parser.add_argument('--num_workers', type=int, default=4)

    # Sampling related arguments
    parser.add_argument('--sample_id', type=str)
    parser.add_argument('--sample_non_ema', type=str2bool, default=True,
                        help='Whether we use the non-ema version model to sample?')

    # Evaluation related arguments
    parser.add_argument('--eval_iter', type=int, default=0, help='Use which iter to evaluate.')
    parser.add_argument('--keep_all_eval_samples', type=str2bool, default=False)
    parser.add_argument('--keep_best_eval_samples', type=str2bool, default=True)
    parser.add_argument('--eval_repeat_num', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--eval_path', type=str, required=True, help="compare with those images")

    # Optimizing related arguments.
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for generator.")
    parser.add_argument('--d_lr', type=float, default=1e-6, help="Learning rate for discriminator.")  # TODO: d's lr
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--ema_beta', type=float, default=0.999)

    # Loss hyper arguments.
    parser.add_argument('--lambda_adv', type=float, default=1)

    # Step related arguments.
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--eval_every', type=int, default=5000)

    # Log related arguments.
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    parser.add_argument('--save_loss', type=str2bool, default=True)

    # Others
    parser.add_argument('--seed', type=int, default=0, help='Seed for random number generator.')
    parser.add_argument('--cudnn_benchmark', type=str2bool, default=True)
    parser.add_argument('--keep_all_models', type=str2bool, default=False)
    parser.add_argument('--pretrained_models', type=str, nargs='+', default=[],
                        help='The name list of the pretrained models that you used.')

    return parser.parse_args()
