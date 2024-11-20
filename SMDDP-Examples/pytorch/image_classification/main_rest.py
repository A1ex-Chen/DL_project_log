# Copyright (c) 2018-2019, NVIDIA CORPORATION
# Copyright (c) 2017-      Facebook, Inc
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import argparse
import random
from copy import deepcopy
import signal
import os

import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import smdistributed.dataparallel.torch.torch_smddp
import torch.distributed as dist

import image_classification.logger as log

from image_classification.smoothing import LabelSmoothing
from image_classification.mixup import NLLMultiLabelSmooth, MixUpWrapper
from image_classification.dataloaders import DATA_BACKEND_CHOICES, get_pytorch_train_loader, get_pytorch_val_loader, get_syntetic_loader
from image_classification.training import *
from image_classification.utils import *
from image_classification.models import (
    resnet50,
    resnext101_32x4d,
    se_resnext101_32x4d,
    efficientnet_b0,
    efficientnet_b4,
    efficientnet_widese_b0,
    efficientnet_widese_b4,
)
import dllogger









    else:


    if args.static_loss_scale != 1.0:
        if not args.amp:
            print("Warning: if --amp is not used, static_loss_scale will be ignored.")

    if args.optimizer_batch_size < 0:
        batch_size_multiplier = 1
    else:
        tbs = args.world_size * args.batch_size
        args.optimizer_batch_size *= int(args.world_size / 8)
        if args.optimizer_batch_size % tbs != 0:
            print(
                "Warning: simulated batch size {} is not divisible by actual batch size {}".format(
                    args.optimizer_batch_size, tbs
                )
            )
        batch_size_multiplier = int(args.optimizer_batch_size / tbs)
        print("BSM: {}".format(batch_size_multiplier))

    start_epoch = 0
    # optionally resume from a checkpoint
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(
                args.resume, map_location=lambda storage, loc: storage.cuda(args.gpu)
            )
            start_epoch = checkpoint["epoch"]
            best_prec1 = checkpoint["best_prec1"]
            model_state = checkpoint["state_dict"]
            optimizer_state = checkpoint["optimizer"]
            if "state_dict_ema" in checkpoint:
                model_state_ema = checkpoint["state_dict_ema"]
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
            if start_epoch >= args.epochs:
                print(
                    f"Launched training for {args.epochs}, checkpoint already run {start_epoch}"
                )
                exit(1)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            model_state = None
            model_state_ema = None
            optimizer_state = None
    else:
        model_state = None
        model_state_ema = None
        optimizer_state = None

    loss = nn.CrossEntropyLoss
    if args.mixup > 0.0:
        loss = lambda: NLLMultiLabelSmooth(args.label_smoothing)
    elif args.label_smoothing > 0.0:
        loss = lambda: LabelSmoothing(args.label_smoothing)

    memory_format = (
        torch.channels_last if args.memory_format == "nhwc" else torch.contiguous_format
    )
    model = model_arch(
        **{
            k: v
            if k != "pretrained"
            else v and (not args.distributed or dist.get_rank() == 0)
            for k, v in model_args.__dict__.items()
        }
    )

    image_size = (
        args.image_size
        if args.image_size is not None
        else model.arch.default_image_size
    )
    model_and_loss = ModelAndLoss(model, loss, cuda=True, memory_format=memory_format)
    if args.use_ema is not None:
        model_ema = deepcopy(model_and_loss)
        ema = EMA(args.use_ema)
    else:
        model_ema = None
        ema = None

    # Create data loaders and optimizers as needed
    if args.data_backend == "pytorch":
        get_train_loader = get_pytorch_train_loader
        get_val_loader = get_pytorch_val_loader
    # elif args.data_backend == "dali-gpu":
    #     get_train_loader = get_dali_train_loader(dali_cpu=False)
    #     get_val_loader = get_dali_val_loader()
    # elif args.data_backend == "dali-cpu":
    #     get_train_loader = get_dali_train_loader(dali_cpu=True)
    #     get_val_loader = get_dali_val_loader()
    elif args.data_backend == "syntetic":
        get_val_loader = get_syntetic_loader
        get_train_loader = get_syntetic_loader
    else:
        print("Bad databackend picked")
        exit(1)

    train_loader, train_loader_len = get_train_loader(
        args.data,
        image_size,
        args.batch_size,
        model_args.num_classes,
        args.mixup > 0.0,
        interpolation = args.interpolation,
        augmentation=args.augmentation,
        start_epoch=start_epoch,
        workers=args.workers,
        memory_format=memory_format,
    )
    if args.mixup != 0.0:
        train_loader = MixUpWrapper(args.mixup, train_loader)

    val_loader, val_loader_len = get_val_loader(
        args.data,
        image_size,
        args.batch_size,
        model_args.num_classes,
        False,
        interpolation = args.interpolation,
        workers=args.workers,
        memory_format=memory_format,
    )

    if not dist.is_initialized() or dist.get_rank() == 0:
        logger = log.Logger(
            args.print_freq,
            [
                dllogger.StdOutBackend(
                    dllogger.Verbosity.DEFAULT, step_format=log.format_step
                ),
                dllogger.JSONStreamBackend(
                    dllogger.Verbosity.VERBOSE,
                    os.path.join(args.workspace, args.raport_file),
                ),
            ],
            start_epoch=start_epoch - 1,
        )

    else:
        logger = log.Logger(args.print_freq, [], start_epoch=start_epoch - 1)

    logger.log_parameter(args.__dict__, verbosity=dllogger.Verbosity.DEFAULT)
    logger.log_parameter(
        {f"model.{k}": v for k, v in model_args.__dict__.items()},
        verbosity=dllogger.Verbosity.DEFAULT,
    )

    optimizer = get_optimizer(
        list(model_and_loss.model.named_parameters()),
        args.lr,
        args=args,
        state=optimizer_state,
    )

    if args.lr_schedule == "step":
        lr_policy = lr_step_policy(
            args.lr, [30, 60, 80], 0.1, args.warmup, logger=logger
        )
    elif args.lr_schedule == "cosine":
        lr_policy = lr_cosine_policy(
            args.lr, args.warmup, args.epochs, end_lr=args.end_lr, logger=logger
        )
    elif args.lr_schedule == "linear":
        lr_policy = lr_linear_policy(args.lr, args.warmup, args.epochs, logger=logger)

    scaler = torch.cuda.amp.GradScaler(
        init_scale=args.static_loss_scale,
        growth_factor=2,
        backoff_factor=0.5,
        growth_interval=100 if args.dynamic_loss_scale else 1000000000,
        enabled=args.amp,
    )

    if args.distributed:
        model_and_loss.distributed(args.gpu)

    model_and_loss.load_model_state(model_state)
    if (ema is not None) and (model_state_ema is not None):
        print("load ema")
        ema.load_state_dict(model_state_ema)

    return (model_and_loss, optimizer, lr_policy, scaler, train_loader, val_loader, logger, ema, model_ema,
            train_loader_len, batch_size_multiplier, start_epoch)


def main(args, model_args, model_arch):
    global best_prec1
    best_prec1 = 0

    model_and_loss, optimizer, lr_policy, scaler, train_loader, val_loader, logger, ema, model_ema, train_loader_len, \
        batch_size_multiplier, start_epoch = prepare_for_training(args, model_args, model_arch)

    exp_start_time = time.time()
    train_loop(
        model_and_loss,
        optimizer,
        scaler,
        lr_policy,
        train_loader,
        val_loader,
        logger,
        should_backup_checkpoint(args),
        ema=ema,
        model_ema=model_ema,
        steps_per_epoch=train_loader_len,
        use_amp=args.amp,
        batch_size_multiplier=batch_size_multiplier,
        start_epoch=start_epoch,
        end_epoch=min((start_epoch + args.run_epochs), args.epochs)
        if args.run_epochs != -1
        else args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        best_prec1=best_prec1,
        prof=args.prof,
        skip_training=args.evaluate,
        skip_validation=args.training_only,
        save_checkpoints=args.save_checkpoints and not args.evaluate,
        checkpoint_dir=args.workspace,
        checkpoint_filename=args.checkpoint_filename,
    )
    exp_duration = time.time() - exp_start_time
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("Experiment ended")
        print("Total training time: {:.2f} secs".format(exp_duration))
        logger.end()


if __name__ == "__main__":

    epilog = [
        "Based on the architecture picked by --arch flag, you may use the following options:\n"
    ]
    for model, ep in available_models().items():
        model_help = "\n".join(ep.parser().format_help().split("\n")[2:])
        epilog.append(model_help)
    parser = argparse.ArgumentParser(
        description="PyTorch ImageNet Training",
        epilog="\n".join(epilog),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    add_parser_arguments(parser)

    args, rest = parser.parse_known_args()
    
    model_arch = available_models()[args.arch]
    model_args, rest = model_arch.parser().parse_known_args(rest)
    print(model_args)

    assert len(rest) == 0, f"Unknown args passed: {rest}"

    cudnn.benchmark = True

    main(args, model_args, model_arch)