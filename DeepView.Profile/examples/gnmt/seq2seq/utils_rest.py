import logging.config
import os
import random
import sys
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.init as init
import torch.utils.collect_env


















@contextmanager


@contextmanager












class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self, skip_first=True):
        self.reset()
        self.skip = skip_first

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val

        if self.skip:
            self.skip = False
        else:
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

    def reduce(self, op):
        """
        Reduces average value over all workers.

        :param op: 'sum' or 'mean', reduction operator
        """
        if op not in ('sum', 'mean'):
            raise NotImplementedError

        distributed = (get_world_size() > 1)
        if distributed:
            # Backward/forward compatibility around
            # https://github.com/pytorch/pytorch/commit/540ef9b1fc5506369a48491af8a285a686689b36 and
            # https://github.com/pytorch/pytorch/commit/044d00516ccd6572c0d6ab6d54587155b02a3b86
            # To accomodate change in Pytorch's distributed API
            if hasattr(dist, "get_backend"):
                _backend = dist.get_backend()
                if hasattr(dist, "DistBackend"):
                    backend_enum_holder = dist.DistBackend
                else:
                    backend_enum_holder = dist.Backend
            else:
                _backend = dist._backend
                backend_enum_holder = dist.dist_backend

            cuda = _backend == backend_enum_holder.NCCL

            if cuda:
                avg = torch.cuda.FloatTensor([self.avg])
                _sum = torch.cuda.FloatTensor([self.sum])
            else:
                avg = torch.FloatTensor([self.avg])
                _sum = torch.FloatTensor([self.sum])

            _reduce_op = dist.reduce_op if hasattr(dist, "reduce_op") else dist.ReduceOp
            dist.all_reduce(avg, op=_reduce_op.SUM)
            dist.all_reduce(_sum, op=_reduce_op.SUM)
            self.avg = avg.item()
            self.sum = _sum.item()

            if op == 'mean':
                self.avg /= get_world_size()
                self.sum /= get_world_size()







    rank = get_rank()
    rank_filter = RankFilter(rank)

    logging_format = "%(asctime)s - %(levelname)s - %(rank)s - %(message)s"
    logging.basicConfig(level=logging.DEBUG,
                        format=logging_format,
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode='w')
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(rank)s: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.getLogger('').addFilter(rank_filter)


def set_device(cuda, local_rank):
    """
    Sets device based on local_rank and returns instance of torch.device.

    :param cuda: if True: use cuda
    :param local_rank: local rank of the worker
    """
    if cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def init_distributed(cuda):
    """
    Initializes distributed backend.

    :param cuda: (bool) if True initializes nccl backend, if False initializes
        gloo backend
    """
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    distributed = (world_size > 1)
    if distributed:
        backend = 'nccl' if cuda else 'gloo'
        dist.init_process_group(backend=backend,
                                init_method='env://')
        assert dist.is_initialized()
    return distributed


def log_env_info():
    """
    Prints information about execution environment.
    """
    logging.info('Collecting environment information...')
    env_info = torch.utils.collect_env.get_pretty_env_info()
    logging.info(f'{env_info}')


def pad_vocabulary(math):
    if math == 'fp16':
        pad_vocab = 8
    elif math == 'fp32':
        pad_vocab = 1
    return pad_vocab


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self, skip_first=True):
        self.reset()
        self.skip = skip_first

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val

        if self.skip:
            self.skip = False
        else:
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

    def reduce(self, op):
        """
        Reduces average value over all workers.

        :param op: 'sum' or 'mean', reduction operator
        """
        if op not in ('sum', 'mean'):
            raise NotImplementedError

        distributed = (get_world_size() > 1)
        if distributed:
            # Backward/forward compatibility around
            # https://github.com/pytorch/pytorch/commit/540ef9b1fc5506369a48491af8a285a686689b36 and
            # https://github.com/pytorch/pytorch/commit/044d00516ccd6572c0d6ab6d54587155b02a3b86
            # To accomodate change in Pytorch's distributed API
            if hasattr(dist, "get_backend"):
                _backend = dist.get_backend()
                if hasattr(dist, "DistBackend"):
                    backend_enum_holder = dist.DistBackend
                else:
                    backend_enum_holder = dist.Backend
            else:
                _backend = dist._backend
                backend_enum_holder = dist.dist_backend

            cuda = _backend == backend_enum_holder.NCCL

            if cuda:
                avg = torch.cuda.FloatTensor([self.avg])
                _sum = torch.cuda.FloatTensor([self.sum])
            else:
                avg = torch.FloatTensor([self.avg])
                _sum = torch.FloatTensor([self.sum])

            _reduce_op = dist.reduce_op if hasattr(dist, "reduce_op") else dist.ReduceOp
            dist.all_reduce(avg, op=_reduce_op.SUM)
            dist.all_reduce(_sum, op=_reduce_op.SUM)
            self.avg = avg.item()
            self.sum = _sum.item()

            if op == 'mean':
                self.avg /= get_world_size()
                self.sum /= get_world_size()


def debug_tensor(tensor, name):
    """
    Simple utility which helps with debugging.
    Takes a tensor and outputs: min, max, avg, std, number of NaNs, number of
    INFs.

    :param tensor: torch tensor
    :param name: name of the tensor (only for logging)
    """
    logging.info(name)
    tensor = tensor.detach().float().cpu().numpy()
    logging.info(f'MIN: {tensor.min()} MAX: {tensor.max()} '
                 f'AVG: {tensor.mean()} STD: {tensor.std()} '
                 f'NAN: {np.isnan(tensor).sum()} INF: {np.isinf(tensor).sum()}')