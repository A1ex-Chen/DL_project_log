# coding: utf-8

# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
# Uncomment to run with pcluster steps
# os.chdir('/workspace/transformer-xl/pytorch')

import argparse
import functools
import itertools
import logging
import math
import shutil
import sys
import time
import warnings

import dllogger
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
try:
    from apex import amp
except ModuleNotFoundError:
    warnings.warn('APEX AMP is unavailable')
try:
    import pyprof
except ModuleNotFoundError:
    warnings.warn('PyProf is unavailable')

#from torch.nn.parallel import DistributedDataParallel
import smdistributed.dataparallel.torch.distributed as dist
from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel

if not dist.is_initialized():
    dist.init_process_group()

import lamb
import utils
from data_utils import get_lm_corpus
from mem_transformer import MemTransformerLM
from utils.data_parallel import BalancedDataParallel
from utils.exp_utils import AverageMeter
from utils.exp_utils import TimeoutHandler
from utils.exp_utils import benchmark
from utils.exp_utils import create_exp_dir
from utils.exp_utils import l2_promote
from utils.exp_utils import log_env_info
from utils.exp_utils import register_ignoring_timeout_handler
























        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        if args.sample_softmax > 0 and optimizer_sparse is not None:
            scheduler_sparse = optim.lr_scheduler.LambdaLR(
                optimizer_sparse,
                lr_lambda=lr_lambda
                )
        else:
            scheduler_sparse = None
    elif args.scheduler == 'dev_perf':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=args.decay_rate, patience=args.patience,
            min_lr=args.lr_min,
            )
        if args.sample_softmax > 0 and optimizer_sparse is not None:
            scheduler_sparse = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_sparse, factor=args.decay_rate, patience=args.patience,
                min_lr=args.lr_min,
                )
        else:
            scheduler_sparse = None
    elif args.scheduler == 'constant':
        pass

    logging.info('=' * 100)
    for k, v in args.__dict__.items():
        logging.info('    - {} : {}'.format(k, v))
    logging.info('=' * 100)
    logging.info('#params = {}'.format(args.n_all_param))
    logging.info('#non emb params = {}'.format(args.n_nonemb_param))

    train_step = 0
    start_epoch = 1
    last_batch = 0
    last_iter = 0
    best_val_loss = None

    if args.restart:
        try:
            checkpoint = load_checkpoint(args.restart)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            scheduler.load_state_dict(checkpoint['scheduler_state'])
            if args.fp16:
                if args.amp == 'pytorch':
                    scaler.load_state_dict(checkpoint['amp_state'])
                elif args.amp == 'apex':
                    amp.load_state_dict(checkpoint['amp_state'])
            train_step = checkpoint['train_step']
            start_epoch = checkpoint['epoch']
            last_batch = checkpoint['batch']
            last_iter = checkpoint['last_iter']
            best_val_loss = checkpoint['best_val_loss']

            if train_step >= args.max_step:
                logging.info(f'Loaded checkpoint after {train_step} steps, but '
                             f'this run was scheduled for a total of '
                             f'{args.max_step} steps, exiting')
                sys.exit(1)

            model.apply(functools.partial(update_dropout, args=args))
            model.apply(functools.partial(update_dropatt, args=args))
        except FileNotFoundError:
            logging.info(f'Could not load checkpoint from {args.restart}, '
                         f'starting training from random init')

    meters = {}
    warmup = args.mem_len // args.tgt_len + 2
    meters['train_throughput'] = AverageMeter(warmup=warmup)
    ###########################################################################
    # Train
    ###########################################################################
    # Loop over epochs.
    # At any point you can hit Ctrl + C to break out of training early.
    start_time = time.time()
    with torch.autograd.profiler.emit_nvtx(enabled=args.profile):
        with TimeoutHandler() as timeout_handler:
            try:
                for epoch in itertools.count(start=start_epoch):
                    if args.roll:
                        tr_iter.roll(seed=args.seed + epoch)
                    train_step, best_val_loss = train(
                        tr_iter, va_iter, model, para_model, model_config,
                        optimizer, optimizer_sparse, scheduler,
                        scheduler_sparse, scaler, vocab, epoch, last_batch,
                        last_iter, train_step, best_val_loss, meters,
                        timeout_handler, device, args
                        )

                    last_batch = 0
                    last_iter = 0

                    if train_step == args.max_step:
                        logging.info('-' * 100)
                        logging.info('End of training')
                        break
            except KeyboardInterrupt:
                logging.info('-' * 100)
                logging.info('Exiting from training early')
    elapsed = time.time() - start_time

    ###########################################################################
    # Test
    ###########################################################################
    summary = {}
    test_path = os.path.join(args.work_dir, 'checkpoint_best.pt')
    if not args.debug and not args.no_eval and os.path.exists(test_path):
        # Load the best saved model.
        checkpoint = load_checkpoint(test_path)
        model.load_state_dict(checkpoint['model_state'])

        # Run on test data.
        test_start_time = time.time()
        with torch.autograd.profiler.emit_nvtx(enabled=args.profile):
            test_loss = evaluate(te_iter, model, args)
            test_loss = utils.distributed.all_reduce_item(test_loss, 'mean')
        test_elapsed = time.time() - test_start_time

        logging.info('=' * 100)
        if args.dataset in ['enwik8', 'text8']:
            logging.info('| End of training | test time: {:5.2f}s | test loss {:5.2f} | test bpc {:9.5f}'.format(
                test_elapsed, test_loss, test_loss / math.log(2)))
        else:
            logging.info('| End of training | test time: {:5.2f}s | test loss {:5.2f} | test ppl {:9.3f}'.format(
                test_elapsed, test_loss, math.exp(test_loss)))
        logging.info('=' * 100)

        summary.update({
            'test_elapsed': test_elapsed,
            'test_loss': test_loss,
            })

        if args.dataset in ['enwik8', 'text8']:
            summary['test_bits_per_character'] = test_loss / math.log(2)
        else:
            summary['test_perplexity'] = math.exp(test_loss)

    logging.info(f'Training time: {(elapsed / 60):.2f} minutes')
    logging.info(f'Training throughput: {meters["train_throughput"].avg:.2f} tok/s')

    if best_val_loss:
        val_perplexity = math.exp(best_val_loss)
    else:
        val_perplexity = None

    summary.update({
        'train_throughput': meters['train_throughput'].avg,
        'train_elapsed': elapsed / 60,
        'valid_loss': best_val_loss,
        'valid_perplexity': val_perplexity,
        })
    dllogger.log(step=tuple(), data=summary)

    passed = benchmark(
        target_perplexity=args.target_perplexity,
        test_perplexity=val_perplexity,
        target_throughput=args.target_throughput,
        test_throughput=meters['train_throughput'].avg
        )
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    # Disable profiling executor
    try:
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
    except AttributeError:
        pass

    # Before we do anything with models, we want to ensure that we get fp16
    # execution of torch.einsum in APEX AMP.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations.
    # Note that running `--apex_amp_opt_level O2` will remove the need for this
    # code, but it is still valid.
    if 'apex' in sys.modules:
        amp.register_half_function(torch, 'einsum')

    main()