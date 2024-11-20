# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import os
import random
import time

# smddp: fix for egg cannot extract issue
os.environ['PYTHON_EGG_CACHE'] = '/tmp'
# smddp: fix for RuntimeError: cannot cache function '__shear_dense': no locator available for file '/opt/conda/lib/python3.6/site-packages/librosa/util/utils.py'
os.environ[ 'NUMBA_CACHE_DIR' ] = '/tmp/'
os.environ["OMP_NUM_THREADS"] = str(1)

import torch
import multiprocessing
import numpy as np
#import torch.distributed as dist
from apex import amp
from torch.cuda.amp import GradScaler
from apex.optimizers import FusedLAMB
#from apex.parallel import DistributedDataParallel
from apex.contrib.optimizers.distributed_fused_lamb import DistributedFusedLAMB
import amp_C
import math

# smddp:
import smdistributed.dataparallel.torch.distributed as dist
from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel
if not dist.is_initialized():
    print('Initiating process group!')
    dist.init_process_group()

from common import helpers
from common.data.dali import sampler as dali_sampler
from common.data.dali.data_loader import DaliDataLoader
from common.data.text import Tokenizer
from common.data import features
from common.helpers import (Checkpointer, greedy_wer, num_weights, print_once,
                            process_evaluation_epoch, Preproc)
from common.optimizers import lr_policy
from common.tb_dllogger import flush_log, init_log, log
from rnnt import config
from rnnt.decoder import RNNTGreedyDecoder
from rnnt.loss import RNNTLoss
from rnnt.loss import apexTransducerLoss
from rnnt.model import RNNT
from rnnt.rnnt_graph import RNNTGraph

from mlperf import logging

# smddp: nvtx can be useful for profiling
#import nvtx

# TODO Eval batch size



@torch.no_grad()






@torch.no_grad()


# smddp: nvtx decorator
#@nvtx.annotate("train step", color="purple")


    if args.dist_lamb:
        optimizer_warm_up()

    logging.log_end(logging.constants.INIT_STOP)
    if multi_gpu:
        dist.barrier()
    logging.log_start(logging.constants.RUN_START)
    if multi_gpu:
        dist.barrier()

    if args.pre_sort_for_seq_split and not args.vectorized_sampler:
        raise NotImplementedError("Pre sort only works with vectorized sampler for now")
    logging.log_event(logging.constants.DATA_TRAIN_NUM_BUCKETS, value=args.num_buckets)

    if args.num_buckets is not None:
        if args.vectorized_sampler:
            builder = dali_sampler.VectorizedBucketingSampler
        else:
            builder = dali_sampler.BucketingSampler

        train_sampler = builder(
            train_dataset_kw,
            args.num_buckets,
            batch_size,
            world_size,
            args.epochs,
            sampler_seed,
            args.dist_sampler,
            args.pre_sort_for_seq_split
        )
    else:
        train_sampler = dali_sampler.SimpleSampler(train_dataset_kw)

    eval_sampler = dali_sampler.SimpleSampler(val_dataset_kw)

    train_sampler.sample(   file_names=args.train_manifests, 
                            in_mem_file_list=args.in_mem_file_list,
                            tokenized_transcript=args.tokenized_transcript)
    eval_sampler.sample(file_names=args.val_manifests, 
                        in_mem_file_list=args.in_mem_file_list,
                        tokenized_transcript=args.tokenized_transcript)


    # Setup DALI pipeline
    if args.synthetic_audio_seq_len is None and args.synthetic_text_seq_len is None:
        synthetic_seq_len = None
    elif args.synthetic_audio_seq_len is not None and args.synthetic_text_seq_len is not None:
        synthetic_seq_len = [args.synthetic_audio_seq_len, args.synthetic_text_seq_len]
    else:
        raise Exception("synthetic seq length for both text and audio need to be specified")
    train_loader = DaliDataLoader(gpu_id=args.local_rank,
                                  dataset_path=args.dataset_dir,
                                  config_data=train_dataset_kw,
                                  config_features=train_features_kw,
                                  json_names=args.train_manifests,
                                  batch_size=batch_size,
                                  sampler=train_sampler,
                                  grad_accumulation_steps=args.grad_accumulation_steps,
                                  pipeline_type="train",
                                  device_type=args.dali_device,
                                  tokenizer=tokenizer,
                                  num_threads=args.data_cpu_threads,
                                  synthetic_seq_len=synthetic_seq_len,
                                  seed=dali_seed,
                                  in_mem_file_list=args.in_mem_file_list,
                                  enable_prefetch=args.enable_prefetch,
                                  tokenized_transcript=args.tokenized_transcript,
                                  preproc=train_preproc,
                                  min_seq_split_len=args.min_seq_split_len,
                                  pre_sort=args.pre_sort_for_seq_split,
                                  jit_tensor_formation=args.jit_tensor_formation,
                                  dont_use_mmap=args.dali_dont_use_mmap)


    val_loader = DaliDataLoader(gpu_id=args.local_rank,
                                dataset_path=args.dataset_dir,
                                config_data=val_dataset_kw,
                                config_features=val_features_kw,
                                json_names=args.val_manifests,
                                batch_size=args.val_batch_size,
                                sampler=eval_sampler,
                                pipeline_type="val",
                                device_type=args.dali_device,
                                tokenizer=tokenizer,
                                num_threads=args.data_cpu_threads,
                                seed=dali_seed,
                                tokenized_transcript=args.tokenized_transcript,
                                in_mem_file_list=args.in_mem_file_list,
                                dont_use_mmap=args.dali_dont_use_mmap)


    steps_per_epoch = len(train_loader) // args.grad_accumulation_steps

    logging.log_event(logging.constants.TRAIN_SAMPLES, value=train_loader.dataset_size)
    logging.log_event(logging.constants.EVAL_SAMPLES, value=val_loader.dataset_size)

    logging.log_event(logging.constants.OPT_NAME, value='lamb')
    logging.log_event(logging.constants.OPT_BASE_LR, value=args.lr)
    logging.log_event(logging.constants.OPT_LAMB_EPSILON, value=opt_eps)
    logging.log_event(logging.constants.OPT_LAMB_LR_DECAY_POLY_POWER, value=args.lr_exp_gamma)
    logging.log_event(logging.constants.OPT_LR_WARMUP_EPOCHS, value=args.warmup_epochs)
    logging.log_event(logging.constants.OPT_LAMB_LR_HOLD_EPOCHS, value=args.hold_epochs)
    logging.log_event(logging.constants.OPT_LAMB_BETA_1, value=args.beta1)
    logging.log_event(logging.constants.OPT_LAMB_BETA_2, value=args.beta2)
    logging.log_event(logging.constants.OPT_GRADIENT_CLIP_NORM, value=args.clip_norm)
    logging.log_event(logging.constants.OPT_LR_ALT_DECAY_FUNC, value=True)
    logging.log_event(logging.constants.OPT_LR_ALT_WARMUP_FUNC, value=True)
    logging.log_event(logging.constants.OPT_LAMB_LR_MIN, value=args.min_lr)
    logging.log_event(logging.constants.OPT_WEIGHT_DECAY, value=args.weight_decay)


    # load checkpoint
    meta = {'best_wer': 10**6, 'start_epoch': 0}
    checkpointer = Checkpointer(args.output_dir, 'RNN-T',
                                args.keep_milestones, use_amp=True)
    if args.resume:
        args.ckpt = checkpointer.last_checkpoint() or args.ckpt

    if args.ckpt is not None:
        checkpointer.load(args.ckpt, model, ema_model, optimizer, meta)


    start_epoch = meta['start_epoch']
    best_wer = meta['best_wer']
    last_wer = meta['best_wer']
    epoch = 1
    step = start_epoch * steps_per_epoch + 1

    # training loop
    model.train()
    if args.multi_tensor_ema == True:
        ema_model_weight_list, model_weight_list, overflow_buf_for_ema = init_multi_tensor_ema(optimizer, model, ema_model, args.ema_update_type)

    copy_stream = torch.cuda.Stream()
    pred_stream = torch.cuda.Stream()

    if not train_specaugm_kw:
        train_specaugm = torch.nn.Identity()
    elif not args.vectorized_sa:
        train_specaugm = features.SpecAugment(optim_level=args.amp_level, **train_specaugm_kw)
    else:
        train_specaugm = features.VectorizedSpecAugment(optim_level=args.amp_level, **train_specaugm_kw)
    train_augmentations = torch.nn.Sequential(
        train_specaugm,
        features.FrameSplicing(optim_level=args.amp_level, **train_splicing_kw),
        features.FillPadding(optim_level=args.amp_level, ),
        features.PadAlign(optim_level=args.amp_level, **train_padalign_kw),
        PermuteAudio(),
    )

    if not val_specaugm_kw:
        val_specaugm = torch.nn.Identity()
    elif not args.vectorized_sa:
        val_specaugm = features.SpecAugment(optim_level=args.amp_level, **val_specaugm_kw)
    else:
        val_specaugm = features.VectorizedSpecAugment(optim_level=args.amp_level, **val_specaugm_kw)
    val_augmentations = torch.nn.Sequential(
        val_specaugm,
        features.FrameSplicing(optim_level=args.amp_level, **val_splicing_kw),
        features.FillPadding(optim_level=args.amp_level, ),
        features.PadAlign(optim_level=args.amp_level, **val_padalign_kw),
        PermuteAudio(),
    )

    train_feat_proc = train_augmentations
    val_feat_proc   = val_augmentations

    train_feat_proc.cuda()
    val_feat_proc.cuda()

    train_preproc = Preproc(train_feat_proc, args.dist_lamb, args.apex_transducer_joint, args.batch_split_factor, cfg)


    # graphing
    if args.num_cg > 0:
        if not args.dist_lamb:
            raise NotImplementedError("Currently CUDA graph training only works with dist LAMB")
        if args.batch_split_factor != 1:
            raise NotImplementedError("Currently CUDA graph training does not work with batch split")

        max_seq_len = math.ceil(train_preproc.audio_duration_to_seq_len(
                                                    cfg['input_train']['audio_dataset']['max_duration'], 
                                                    after_subsampling=True,
                                                    after_stack_time=False
                                                    ) 
                        * cfg["input_train"]["audio_dataset"]["speed_perturbation"]["max_rate"])

        print_once(f'Graph with max_seq_len of %d' % max_seq_len)
        rnnt_graph = RNNTGraph(model, rnnt_config, batch_size, max_seq_len, args.max_txt_len, args.num_cg)
        rnnt_graph.capture_graph()
    else:
        rnnt_graph = None

    # capture CG for eval
    if type(args.batch_eval_mode) == str and args.batch_eval_mode.startswith("cg"):
        max_seq_len = train_preproc.audio_duration_to_seq_len(  args.max_eval_sample_duration, 
                                                                after_subsampling=True, 
                                                                after_stack_time=True) 
        dict_meta_data = {"batch": args.val_batch_size, "max_feat_len": max_seq_len}
        greedy_decoder.capture_cg_for_eval(ema_model, dict_meta_data)

    # warm up optimizer
    def optimizer_warm_up():
        WARMUP_LEN = 8
        feats = torch.ones(WARMUP_LEN , batch_size, rnnt_config["in_feats"], dtype=torch.float16, device='cuda')
        feat_lens = torch.ones(batch_size, dtype=torch.int32, device='cuda') * WARMUP_LEN 
        txt = torch.ones(batch_size, WARMUP_LEN , dtype=torch.int64, device='cuda')
        txt_lens = torch.ones(batch_size, dtype=torch.int32, device='cuda') * WARMUP_LEN 
        dict_meta_data = train_preproc.get_packing_meta_data(feats.size(0), feat_lens, txt_lens)
        log_probs, log_prob_lens = model(feats, feat_lens, txt, txt_lens, dict_meta_data)
        loss = loss_fn(log_probs, log_prob_lens, txt, txt_lens, dict_meta_data)
        loss /= args.grad_accumulation_steps
        del log_probs, log_prob_lens

        assert not torch.isnan(loss).any(), "should not have happened"
        if args.dist_lamb:
            optimizer._lazy_init_stage1()
            grad_scaler.scale(loss).backward()
            optimizer._lazy_init_stage2()
        else:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        optimizer.zero_grad()   # Don't really want to do the update
        if args.dist_lamb:
            optimizer.complete_reductions()
            optimizer.set_global_scale(grad_scaler._get_scale_async())
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            optimizer.step()

    if args.dist_lamb:
        optimizer_warm_up()

    logging.log_end(logging.constants.INIT_STOP)
    if multi_gpu:
        dist.barrier()
    logging.log_start(logging.constants.RUN_START)
    if multi_gpu:
        dist.barrier()

    if args.pre_sort_for_seq_split and not args.vectorized_sampler:
        raise NotImplementedError("Pre sort only works with vectorized sampler for now")
    logging.log_event(logging.constants.DATA_TRAIN_NUM_BUCKETS, value=args.num_buckets)

    if args.num_buckets is not None:
        if args.vectorized_sampler:
            builder = dali_sampler.VectorizedBucketingSampler
        else:
            builder = dali_sampler.BucketingSampler

        train_sampler = builder(
            train_dataset_kw,
            args.num_buckets,
            batch_size,
            world_size,
            args.epochs,
            sampler_seed,
            args.dist_sampler,
            args.pre_sort_for_seq_split
        )
    else:
        train_sampler = dali_sampler.SimpleSampler(train_dataset_kw)

    eval_sampler = dali_sampler.SimpleSampler(val_dataset_kw)

    train_sampler.sample(   file_names=args.train_manifests, 
                            in_mem_file_list=args.in_mem_file_list,
                            tokenized_transcript=args.tokenized_transcript)
    eval_sampler.sample(file_names=args.val_manifests, 
                        in_mem_file_list=args.in_mem_file_list,
                        tokenized_transcript=args.tokenized_transcript)


    # Setup DALI pipeline
    if args.synthetic_audio_seq_len is None and args.synthetic_text_seq_len is None:
        synthetic_seq_len = None
    elif args.synthetic_audio_seq_len is not None and args.synthetic_text_seq_len is not None:
        synthetic_seq_len = [args.synthetic_audio_seq_len, args.synthetic_text_seq_len]
    else:
        raise Exception("synthetic seq length for both text and audio need to be specified")
    train_loader = DaliDataLoader(gpu_id=args.local_rank,
                                  dataset_path=args.dataset_dir,
                                  config_data=train_dataset_kw,
                                  config_features=train_features_kw,
                                  json_names=args.train_manifests,
                                  batch_size=batch_size,
                                  sampler=train_sampler,
                                  grad_accumulation_steps=args.grad_accumulation_steps,
                                  pipeline_type="train",
                                  device_type=args.dali_device,
                                  tokenizer=tokenizer,
                                  num_threads=args.data_cpu_threads,
                                  synthetic_seq_len=synthetic_seq_len,
                                  seed=dali_seed,
                                  in_mem_file_list=args.in_mem_file_list,
                                  enable_prefetch=args.enable_prefetch,
                                  tokenized_transcript=args.tokenized_transcript,
                                  preproc=train_preproc,
                                  min_seq_split_len=args.min_seq_split_len,
                                  pre_sort=args.pre_sort_for_seq_split,
                                  jit_tensor_formation=args.jit_tensor_formation,
                                  dont_use_mmap=args.dali_dont_use_mmap)


    val_loader = DaliDataLoader(gpu_id=args.local_rank,
                                dataset_path=args.dataset_dir,
                                config_data=val_dataset_kw,
                                config_features=val_features_kw,
                                json_names=args.val_manifests,
                                batch_size=args.val_batch_size,
                                sampler=eval_sampler,
                                pipeline_type="val",
                                device_type=args.dali_device,
                                tokenizer=tokenizer,
                                num_threads=args.data_cpu_threads,
                                seed=dali_seed,
                                tokenized_transcript=args.tokenized_transcript,
                                in_mem_file_list=args.in_mem_file_list,
                                dont_use_mmap=args.dali_dont_use_mmap)


    steps_per_epoch = len(train_loader) // args.grad_accumulation_steps

    logging.log_event(logging.constants.TRAIN_SAMPLES, value=train_loader.dataset_size)
    logging.log_event(logging.constants.EVAL_SAMPLES, value=val_loader.dataset_size)

    logging.log_event(logging.constants.OPT_NAME, value='lamb')
    logging.log_event(logging.constants.OPT_BASE_LR, value=args.lr)
    logging.log_event(logging.constants.OPT_LAMB_EPSILON, value=opt_eps)
    logging.log_event(logging.constants.OPT_LAMB_LR_DECAY_POLY_POWER, value=args.lr_exp_gamma)
    logging.log_event(logging.constants.OPT_LR_WARMUP_EPOCHS, value=args.warmup_epochs)
    logging.log_event(logging.constants.OPT_LAMB_LR_HOLD_EPOCHS, value=args.hold_epochs)
    logging.log_event(logging.constants.OPT_LAMB_BETA_1, value=args.beta1)
    logging.log_event(logging.constants.OPT_LAMB_BETA_2, value=args.beta2)
    logging.log_event(logging.constants.OPT_GRADIENT_CLIP_NORM, value=args.clip_norm)
    logging.log_event(logging.constants.OPT_LR_ALT_DECAY_FUNC, value=True)
    logging.log_event(logging.constants.OPT_LR_ALT_WARMUP_FUNC, value=True)
    logging.log_event(logging.constants.OPT_LAMB_LR_MIN, value=args.min_lr)
    logging.log_event(logging.constants.OPT_WEIGHT_DECAY, value=args.weight_decay)


    # load checkpoint
    meta = {'best_wer': 10**6, 'start_epoch': 0}
    checkpointer = Checkpointer(args.output_dir, 'RNN-T',
                                args.keep_milestones, use_amp=True)
    if args.resume:
        args.ckpt = checkpointer.last_checkpoint() or args.ckpt

    if args.ckpt is not None:
        checkpointer.load(args.ckpt, model, ema_model, optimizer, meta)


    start_epoch = meta['start_epoch']
    best_wer = meta['best_wer']
    last_wer = meta['best_wer']
    epoch = 1
    step = start_epoch * steps_per_epoch + 1

    # training loop
    model.train()
    if args.multi_tensor_ema == True:
        ema_model_weight_list, model_weight_list, overflow_buf_for_ema = init_multi_tensor_ema(optimizer, model, ema_model, args.ema_update_type)

    copy_stream = torch.cuda.Stream()
    pred_stream = torch.cuda.Stream()
    def buffer_pre_allocation():
        max_seq_len = math.ceil(train_preproc.audio_duration_to_seq_len(
                                                    cfg['input_train']['audio_dataset']['max_duration'], 
                                                    after_subsampling=False,
                                                    after_stack_time=False
                                                    ) 
                        * cfg["input_train"]["audio_dataset"]["speed_perturbation"]["max_rate"])
        max_txt_len = train_loader.data_iterator().max_txt_len
        print_once(f'Pre-allocate buffer with max_seq_len of %d and max_txt_len of %d' % (max_seq_len, max_txt_len))
        audio = torch.ones(batch_size, cfg["input_val"]["filterbank_features"]["n_filt"], max_seq_len, dtype=torch.float32, device='cuda')
        audio_lens = torch.ones(batch_size, dtype=torch.int32, device='cuda') * max_seq_len
        txt = torch.ones(batch_size, max_txt_len, dtype=torch.int64, device='cuda')
        txt_lens = torch.ones(batch_size, dtype=torch.int32, device='cuda') * max_txt_len
        feats, feat_lens = train_feat_proc([audio, audio_lens])
        if args.dist_lamb:
            feats = feats.half()

        meta_data = []
        B_split = batch_size // args.batch_split_factor
        for i in range(args.batch_split_factor):
            meta_data.append(train_preproc.get_packing_meta_data(   feats.size(0), 
                                                                    feat_lens[i*B_split:(i+1)*B_split], 
                                                                    txt_lens[i*B_split:(i+1)*B_split]))

        train_step( model, loss_fn, args, batch_size, feats, feat_lens, txt, txt_lens, optimizer, 
                    grad_scaler, meta_data, None, rnnt_graph, copy_stream, pred_stream)


    if args.buffer_pre_alloc:
        buffer_pre_allocation()

    training_start_time = time.time()
    training_utts = 0
    for epoch in range(start_epoch + 1, args.epochs + 1):

        logging.log_start(logging.constants.BLOCK_START,
                          metadata=dict(first_epoch_num=epoch,
                                        epoch_count=1))
        logging.log_start(logging.constants.EPOCH_START,
                          metadata=dict(epoch_num=epoch))

        epoch_utts = 0
        accumulated_batches = 0
        epoch_start_time = time.time()
        

        if args.enable_prefetch:
            train_loader.data_iterator().prefetch()

        step_start_time = time.time()

        for batch in train_loader:
            if accumulated_batches == 0:
                if not args.dist_lamb:
                    optimizer.zero_grad()
                else:
                    optimizer.zero_grad(set_to_none=True)
                adjust_lr(step, epoch)
                step_utts = 0
                all_feat_lens = []

            if args.enable_prefetch:
                # when prefetch is enabled, train_feat_proc at prefetch time
                feats, feat_lens, txt, txt_lens = batch
                meta_data = train_preproc.meta_data
            else:
                audio, audio_lens, txt, txt_lens = batch
                feats, feat_lens = train_feat_proc([audio, audio_lens])
                if args.dist_lamb:
                    feats = feats.half()
                meta_data = []
                B_split = batch_size // args.batch_split_factor
                for i in range(args.batch_split_factor):
                    meta_data.append(train_preproc.get_packing_meta_data(   feats.size(0), 
                                                                            feat_lens[i*B_split:(i+1)*B_split], 
                                                                            txt_lens[i*B_split:(i+1)*B_split]))
                        
            if args.enable_seq_len_stats:
                all_feat_lens += feat_lens



            loss_item, lr_item = train_step( model, loss_fn, args, batch_size, feats, feat_lens, txt, txt_lens, optimizer, 
                                    grad_scaler, meta_data, train_loader, rnnt_graph, copy_stream, pred_stream)


            step_utts += txt_lens.size(0) * world_size
            epoch_utts += txt_lens.size(0) * world_size
            accumulated_batches += 1
            if accumulated_batches % args.grad_accumulation_steps == 0:

                if args.dist_lamb:
                    optimizer.complete_reductions()
                    optimizer.set_global_scale(grad_scaler._get_scale_async())
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                else:
                    optimizer.step()

                if args.multi_tensor_ema == True:
                    apply_multi_tensor_ema(model_weight_list, ema_model_weight_list, args.ema, overflow_buf_for_ema)
                else:
                    apply_ema(optimizer, ema_model, args.ema)

                if step % args.log_frequency == 0:

                    if args.prediction_frequency is None or step % args.prediction_frequency == 0:
                        preds = greedy_decoder.decode(feats.half(), feat_lens)
                        wer, pred_utt, ref = greedy_wer(
                                preds,
                                txt,
                                txt_lens,
                                tokenizer.detokenize)
                        print_once(f'  Decoded:   {pred_utt[:90]}')
                        print_once(f'  Reference: {ref[:90]}')
                        wer = {'wer': 100 * wer}
                    else:
                        wer = {}

                    step_time = time.time() - step_start_time
                    step_start_time = time.time()
                    dict_log = {'loss': loss_item,
                                 **wer,  # optional entry
                                'throughput': step_utts / step_time,
                                'took': step_time,
                                'lrate': optimizer._lr.item() if args.dist_lamb else optimizer.param_groups[0]['lr']} # TODO: eliminate sync

                    if args.enable_seq_len_stats:
                        dict_log["seq-len-min"] = min(all_feat_lens).item()
                        dict_log["seq-len-max"] = max(all_feat_lens).item()

                    log((epoch, step % steps_per_epoch or steps_per_epoch, steps_per_epoch),
                        step, 'train', dict_log)



                step += 1
                accumulated_batches = 0
                # end of step

        logging.log_end(logging.constants.EPOCH_STOP,
                        metadata=dict(epoch_num=epoch))

        epoch_time = time.time() - epoch_start_time
        log((epoch,), None, 'train_avg', {'throughput': epoch_utts / epoch_time,
                                          'took': epoch_time})
        # logging throughput for dashboard
        logging.log_event(key='throughput', value= epoch_utts / epoch_time)

        if epoch % args.val_frequency == 0:
            wer = evaluate(epoch, step, val_loader, val_feat_proc,
                           tokenizer.detokenize, ema_model, loss_fn,
                           greedy_decoder, args.amp_level)

            last_wer = wer
            if wer < best_wer and epoch >= args.save_best_from:
                checkpointer.save(model, ema_model, optimizer, epoch,
                                  step, best_wer, is_best=True)
                best_wer = wer

        save_this_epoch = (args.save_frequency is not None and epoch % args.save_frequency == 0) \
                       or (epoch in args.keep_milestones)
        if save_this_epoch:
            checkpointer.save(model, ema_model, optimizer, epoch, step, best_wer)

        training_utts += epoch_utts
        logging.log_end(logging.constants.BLOCK_STOP, metadata=dict(first_epoch_num=epoch))

        if last_wer <= args.target:
            logging.log_end(logging.constants.RUN_STOP, metadata={'status': 'success'})
            print_once(f'Finished after {args.epochs_this_job} epochs.')
            break
        if 0 < args.epochs_this_job <= epoch - start_epoch:
            print_once(f'Finished after {args.epochs_this_job} epochs.')
            break
        # end of epoch


    training_time = time.time() - training_start_time
    log((), None, 'train_avg', {'throughput': training_utts / training_time})

    if last_wer > args.target:
        logging.log_end(logging.constants.RUN_STOP, metadata={'status': 'aborted'})

    if epoch == args.epochs:
        # smddp: 
        print('Evaluating the model at the end')
        evaluate(epoch, step, val_loader, val_feat_proc, tokenizer.detokenize,
                 ema_model, loss_fn, greedy_decoder, args.amp_level)

    flush_log()
    if args.save_at_the_end:
        # smddp:
        print('Saving the model at the end')
        checkpointer.save(model, ema_model, optimizer, epoch, step, best_wer)


if __name__ == "__main__":
    main()