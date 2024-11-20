#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import random
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import torch.utils.checkpoint
import transformers
from datasets import load_dataset, load_from_disk
from flax import jax_utils
from flax.core.frozen_dict import unfreeze
from flax.training import train_state
from flax.training.common_utils import shard
from huggingface_hub import create_repo, upload_folder
from PIL import Image, PngImagePlugin
from torch.utils.data import IterableDataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, FlaxCLIPTextModel, set_seed

from diffusers import (
    FlaxAutoencoderKL,
    FlaxControlNetModel,
    FlaxDDPMScheduler,
    FlaxStableDiffusionControlNetPipeline,
    FlaxUNet2DConditionModel,
)
from diffusers.utils import check_min_version, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card


# To prevent an error that occurs when there are abnormally large compressed data chunk in the png image
# see more https://github.com/python-pillow/Pillow/issues/5610
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")

logger = logging.getLogger(__name__)















    image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
        ]
    )


    if jax.process_index() == 0:
        if args.max_train_samples is not None:
            if args.streaming:
                dataset["train"] = dataset["train"].shuffle(seed=args.seed).take(args.max_train_samples)
            else:
                dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        if args.streaming:
            train_dataset = dataset["train"].map(
                preprocess_train,
                batched=True,
                batch_size=batch_size,
                remove_columns=list(dataset["train"].features.keys()),
            )
        else:
            train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([example["input_ids"] for example in examples])

    batch = {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
    }
    batch = {k: v.numpy() for k, v in batch.items()}
    return batch


def get_params_to_save(params):
    return jax.device_get(jax.tree_util.tree_map(lambda x: x[0], params))


def main():
    args = parse_args()

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    # wandb init
    if jax.process_index() == 0 and args.report_to == "wandb":
        wandb.init(
            entity=args.wandb_entity,
            project=args.tracker_project_name,
            job_type="train",
            config=args,
        )

    if args.seed is not None:
        set_seed(args.seed)

    rng = jax.random.PRNGKey(0)

    # Handle the repository creation
    if jax.process_index() == 0:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer and add the placeholder token as a additional special token
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
        )
    else:
        raise NotImplementedError("No tokenizer specified!")

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    total_train_batch_size = args.train_batch_size * jax.local_device_count() * args.gradient_accumulation_steps
    train_dataset = make_train_dataset(args, tokenizer, batch_size=total_train_batch_size)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=not args.streaming,
        collate_fn=collate_fn,
        batch_size=total_train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    weight_dtype = jnp.float32
    if args.mixed_precision == "fp16":
        weight_dtype = jnp.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = jnp.bfloat16

    # Load models and create wrapper for stable diffusion
    text_encoder = FlaxCLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        dtype=weight_dtype,
        revision=args.revision,
        from_pt=args.from_pt,
    )
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        subfolder="vae",
        dtype=weight_dtype,
        from_pt=args.from_pt,
    )
    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        dtype=weight_dtype,
        revision=args.revision,
        from_pt=args.from_pt,
    )

    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet, controlnet_params = FlaxControlNetModel.from_pretrained(
            args.controlnet_model_name_or_path,
            revision=args.controlnet_revision,
            from_pt=args.controlnet_from_pt,
            dtype=jnp.float32,
        )
    else:
        logger.info("Initializing controlnet weights from unet")
        rng, rng_params = jax.random.split(rng)

        controlnet = FlaxControlNetModel(
            in_channels=unet.config.in_channels,
            down_block_types=unet.config.down_block_types,
            only_cross_attention=unet.config.only_cross_attention,
            block_out_channels=unet.config.block_out_channels,
            layers_per_block=unet.config.layers_per_block,
            attention_head_dim=unet.config.attention_head_dim,
            cross_attention_dim=unet.config.cross_attention_dim,
            use_linear_projection=unet.config.use_linear_projection,
            flip_sin_to_cos=unet.config.flip_sin_to_cos,
            freq_shift=unet.config.freq_shift,
        )
        controlnet_params = controlnet.init_weights(rng=rng_params)
        controlnet_params = unfreeze(controlnet_params)
        for key in [
            "conv_in",
            "time_embedding",
            "down_blocks_0",
            "down_blocks_1",
            "down_blocks_2",
            "down_blocks_3",
            "mid_block",
        ]:
            controlnet_params[key] = unet_params[key]

    pipeline, pipeline_params = FlaxStableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        tokenizer=tokenizer,
        controlnet=controlnet,
        safety_checker=None,
        dtype=weight_dtype,
        revision=args.revision,
        from_pt=args.from_pt,
    )
    pipeline_params = jax_utils.replicate(pipeline_params)

    # Optimization
    if args.scale_lr:
        args.learning_rate = args.learning_rate * total_train_batch_size

    constant_scheduler = optax.constant_schedule(args.learning_rate)

    adamw = optax.adamw(
        learning_rate=constant_scheduler,
        b1=args.adam_beta1,
        b2=args.adam_beta2,
        eps=args.adam_epsilon,
        weight_decay=args.adam_weight_decay,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        adamw,
    )

    state = train_state.TrainState.create(apply_fn=controlnet.__call__, params=controlnet_params, tx=optimizer)

    noise_scheduler, noise_scheduler_state = FlaxDDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # Initialize our training
    validation_rng, train_rngs = jax.random.split(rng)
    train_rngs = jax.random.split(train_rngs, jax.local_device_count())



        grad_fn = jax.value_and_grad(compute_loss)

        # get a minibatch (one gradient accumulation slice)


        if args.gradient_accumulation_steps == 1:
            loss, grad, new_train_rng = loss_and_grad(None, train_rng)
        else:
            init_loss_grad_rng = (
                0.0,  # initial value for cumul_loss
                jax.tree_map(jnp.zeros_like, state.params),  # initial value for cumul_grad
                train_rng,  # initial value for train_rng
            )

            def cumul_grad_step(grad_idx, loss_grad_rng):
                cumul_loss, cumul_grad, train_rng = loss_grad_rng
                loss, grad, new_train_rng = loss_and_grad(grad_idx, train_rng)
                cumul_loss, cumul_grad = jax.tree_map(jnp.add, (cumul_loss, cumul_grad), (loss, grad))
                return cumul_loss, cumul_grad, new_train_rng

            loss, grad, new_train_rng = jax.lax.fori_loop(
                0,
                args.gradient_accumulation_steps,
                cumul_grad_step,
                init_loss_grad_rng,
            )
            loss, grad = jax.tree_map(lambda x: x / args.gradient_accumulation_steps, (loss, grad))

        grad = jax.lax.pmean(grad, "batch")

        new_state = state.apply_gradients(grads=grad)

        metrics = {"loss": loss}
        metrics = jax.lax.pmean(metrics, axis_name="batch")


            loss, grad, new_train_rng = jax.lax.fori_loop(
                0,
                args.gradient_accumulation_steps,
                cumul_grad_step,
                init_loss_grad_rng,
            )
            loss, grad = jax.tree_map(lambda x: x / args.gradient_accumulation_steps, (loss, grad))

        grad = jax.lax.pmean(grad, "batch")

        new_state = state.apply_gradients(grads=grad)

        metrics = {"loss": loss}
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        def l2(xs):
            return jnp.sqrt(sum([jnp.vdot(x, x) for x in jax.tree_util.tree_leaves(xs)]))

        metrics["l2_grads"] = l2(jax.tree_util.tree_leaves(grad))

        return new_state, metrics, new_train_rng

    # Create parallel version of the train step
    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))

    # Replicate the train state on each device
    state = jax_utils.replicate(state)
    unet_params = jax_utils.replicate(unet_params)
    text_encoder_params = jax_utils.replicate(text_encoder.params)
    vae_params = jax_utils.replicate(vae_params)

    # Train!
    if args.streaming:
        dataset_length = args.max_train_samples
    else:
        dataset_length = len(train_dataloader)
    num_update_steps_per_epoch = math.ceil(dataset_length / args.gradient_accumulation_steps)

    # Scheduler and math around the number of training steps.
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {args.max_train_samples if args.streaming else len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & distributed) = {total_train_batch_size}")
    logger.info(f"  Total optimization steps = {args.num_train_epochs * num_update_steps_per_epoch}")

    if jax.process_index() == 0 and args.report_to == "wandb":
        wandb.define_metric("*", step_metric="train/step")
        wandb.define_metric("train/step", step_metric="walltime")
        wandb.config.update(
            {
                "num_train_examples": args.max_train_samples if args.streaming else len(train_dataset),
                "total_train_batch_size": total_train_batch_size,
                "total_optimization_step": args.num_train_epochs * num_update_steps_per_epoch,
                "num_devices": jax.device_count(),
                "controlnet_params": sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(state.params)),
            }
        )

    global_step = step0 = 0
    epochs = tqdm(
        range(args.num_train_epochs),
        desc="Epoch ... ",
        position=0,
        disable=jax.process_index() > 0,
    )
    if args.profile_memory:
        jax.profiler.save_device_memory_profile(os.path.join(args.output_dir, "memory_initial.prof"))
    t00 = t0 = time.monotonic()
    for epoch in epochs:
        # ======================== Training ================================

        train_metrics = []
        train_metric = None

        steps_per_epoch = (
            args.max_train_samples // total_train_batch_size
            if args.streaming or args.max_train_samples
            else len(train_dataset) // total_train_batch_size
        )
        train_step_progress_bar = tqdm(
            total=steps_per_epoch,
            desc="Training...",
            position=1,
            leave=False,
            disable=jax.process_index() > 0,
        )
        # train
        for batch in train_dataloader:
            if args.profile_steps and global_step == 1:
                train_metric["loss"].block_until_ready()
                jax.profiler.start_trace(args.output_dir)
            if args.profile_steps and global_step == 1 + args.profile_steps:
                train_metric["loss"].block_until_ready()
                jax.profiler.stop_trace()

            batch = shard(batch)
            with jax.profiler.StepTraceAnnotation("train", step_num=global_step):
                state, train_metric, train_rngs = p_train_step(
                    state, unet_params, text_encoder_params, vae_params, batch, train_rngs
                )
            train_metrics.append(train_metric)

            train_step_progress_bar.update(1)

            global_step += 1
            if global_step >= args.max_train_steps:
                break

            if (
                args.validation_prompt is not None
                and global_step % args.validation_steps == 0
                and jax.process_index() == 0
            ):
                _ = log_validation(
                    pipeline, pipeline_params, state.params, tokenizer, args, validation_rng, weight_dtype
                )

            if global_step % args.logging_steps == 0 and jax.process_index() == 0:
                if args.report_to == "wandb":
                    train_metrics = jax_utils.unreplicate(train_metrics)
                    train_metrics = jax.tree_util.tree_map(lambda *m: jnp.array(m).mean(), *train_metrics)
                    wandb.log(
                        {
                            "walltime": time.monotonic() - t00,
                            "train/step": global_step,
                            "train/epoch": global_step / dataset_length,
                            "train/steps_per_sec": (global_step - step0) / (time.monotonic() - t0),
                            **{f"train/{k}": v for k, v in train_metrics.items()},
                        }
                    )
                t0, step0 = time.monotonic(), global_step
                train_metrics = []
            if global_step % args.checkpointing_steps == 0 and jax.process_index() == 0:
                controlnet.save_pretrained(
                    f"{args.output_dir}/{global_step}",
                    params=get_params_to_save(state.params),
                )

        train_metric = jax_utils.unreplicate(train_metric)
        train_step_progress_bar.close()
        epochs.write(f"Epoch... ({epoch + 1}/{args.num_train_epochs} | Loss: {train_metric['loss']})")

    # Final validation & store model.
    if jax.process_index() == 0:
        if args.validation_prompt is not None:
            if args.profile_validation:
                jax.profiler.start_trace(args.output_dir)
            image_logs = log_validation(
                pipeline, pipeline_params, state.params, tokenizer, args, validation_rng, weight_dtype
            )
            if args.profile_validation:
                jax.profiler.stop_trace()
        else:
            image_logs = None

        controlnet.save_pretrained(
            args.output_dir,
            params=get_params_to_save(state.params),
        )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    if args.profile_memory:
        jax.profiler.save_device_memory_profile(os.path.join(args.output_dir, "memory_final.prof"))
    logger.info("Finished training.")


if __name__ == "__main__":
    main()