# Source: https://github.com/y0ast/pytorch-snippets/blob/main/minimal_cifar/train_cifar.py

import argparse

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from deeplite_torch_zoo import get_dataloaders, get_model
from deeplite_torch_zoo.utils import LOGGER








if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate (default: 0.1)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--milestones', nargs='+', type=int, default=[60, 120, 160],
                        help='Milestone epochs for LR schedule (default: [60, 120, 160])')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum value (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (default: 5e-4)')
    parser.add_argument('--lr_gamma', type=float, default=0.2, help='LR gamma (default: 0.2)')
    parser.add_argument('--workers', type=int, default=8, help='Number of data loading workers (default: 8)')
    parser.add_argument('--model', type=str, default='resnet50', help='Model architecture (default: resnet50)')
    parser.add_argument('--dataset_name', type=str, default='cifar100', help='Name of the dataset (default: cifar100)')
    parser.add_argument('--data_root', type=str, default='./', help='Root directory of the dataset (default: ./)')

    parser.add_argument('--dryrun', action='store_true', help='Dry run mode for testing')

    args = parser.parse_args()
    train(args)