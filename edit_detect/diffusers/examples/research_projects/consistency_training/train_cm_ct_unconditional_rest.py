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
"""Script to train a consistency model from scratch via (improved) consistency training."""

import argparse
import gc
import logging
import math
import os
import shutil
from datetime import timedelta
from pathlib import Path

import accelerate
import datasets
import numpy as np
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import (
    CMStochasticIterativeScheduler,
    ConsistencyModelPipeline,
    UNet2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, resolve_interpolation_mode
from diffusers.utils import is_tensorboard_available, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module


if is_wandb_available():
    import wandb


logger = get_logger(__name__, log_level="INFO")































    logger.info(f"Dataset size: {len(dataset)}")

    dataset.set_transform(transform_images)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )

    # 9. Initialize the learning rate scheduler
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # 10. Prepare for training
    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Function for unwraping if torch.compile() was used in accelerate.


        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # 6. Enable optimizations
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            teacher_unet.enable_xformers_memory_efficient_attention()
            if args.use_ema:
                ema_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.optimizer_type == "radam":
        optimizer_class = torch.optim.RAdam
    elif args.optimizer_type == "adamw":
        # Use 8-bit Adam for lower memory usage or to fine-tune the model for 16GB GPUs
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW
    else:
        raise ValueError(
            f"Optimizer type {args.optimizer_type} is not supported. Currently supported optimizer types are `radam`"
            f" and `adamw`."
        )

    # 7. Initialize the optimizer
    optimizer = optimizer_class(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # 8. Dataset creation and data preprocessing
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            split="train",
        )
    else:
        dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, cache_dir=args.cache_dir, split="train")
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets and DataLoaders creation.
    interpolation_mode = resolve_interpolation_mode(args.interpolation_type)
    augmentations = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=interpolation_mode),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform_images(examples):
        images = [augmentations(image.convert("RGB")) for image in examples[args.dataset_image_column_name]]
        batch_dict = {"images": images}
        if args.class_conditional:
            batch_dict["class_labels"] = examples[args.dataset_class_label_column_name]
        return batch_dict

    logger.info(f"Dataset size: {len(dataset)}")

    dataset.set_transform(transform_images)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )

    # 9. Initialize the learning rate scheduler
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # 10. Prepare for training
    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    def recalculate_num_discretization_step_values(discretization_steps, skip_steps):
        """
        Recalculates all quantities depending on the number of discretization steps N.
        """
        noise_scheduler = CMStochasticIterativeScheduler(
            num_train_timesteps=discretization_steps,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            rho=args.rho,
        )
        current_timesteps = get_karras_sigmas(discretization_steps, args.sigma_min, args.sigma_max, args.rho)
        valid_teacher_timesteps_plus_one = current_timesteps[: len(current_timesteps) - skip_steps + 1]
        # timestep_weights are the unnormalized probabilities of sampling the timestep/noise level at each index
        timestep_weights = get_discretized_lognormal_weights(
            valid_teacher_timesteps_plus_one, p_mean=args.p_mean, p_std=args.p_std
        )
        # timestep_loss_weights is the timestep-dependent loss weighting schedule lambda(sigma_i)
        timestep_loss_weights = get_loss_weighting_schedule(valid_teacher_timesteps_plus_one)

        current_timesteps = current_timesteps.to(accelerator.device)
        timestep_weights = timestep_weights.to(accelerator.device)
        timestep_loss_weights = timestep_loss_weights.to(accelerator.device)

        return noise_scheduler, current_timesteps, timestep_weights, timestep_loss_weights

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Function for unwraping if torch.compile() was used in accelerate.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    # Resolve the c parameter for the Pseudo-Huber loss
    if args.huber_c is None:
        args.huber_c = 0.00054 * args.resolution * math.sqrt(unet.config.in_channels)

    # Get current number of discretization steps N according to our discretization curriculum
    current_discretization_steps = get_discretization_steps(
        initial_global_step,
        args.max_train_steps,
        s_0=args.discretization_s_0,
        s_1=args.discretization_s_1,
        constant=args.constant_discretization_steps,
    )
    current_skip_steps = get_skip_steps(initial_global_step, initial_skip=args.skip_steps)
    if current_skip_steps >= current_discretization_steps:
        raise ValueError(
            f"The current skip steps is {current_skip_steps}, but should be smaller than the current number of"
            f" discretization steps {current_discretization_steps}"
        )
    # Recalculate all quantities depending on the number of discretization steps N
    (
        noise_scheduler,
        current_timesteps,
        timestep_weights,
        timestep_loss_weights,
    ) = recalculate_num_discretization_step_values(current_discretization_steps, current_skip_steps)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # 11. Train!
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            # 1. Get batch of images from dataloader (sample x ~ p_data(x))
            clean_images = batch["images"].to(weight_dtype)
            if args.class_conditional:
                class_labels = batch["class_labels"]
            else:
                class_labels = None
            bsz = clean_images.shape[0]

            # 2. Sample a random timestep for each image according to the noise schedule.
            # Sample random indices i ~ p(i), where p(i) is the dicretized lognormal distribution in the iCT paper
            # NOTE: timestep_indices should be in the range [0, len(current_timesteps) - k - 1] inclusive
            timestep_indices = torch.multinomial(timestep_weights, bsz, replacement=True).long()
            teacher_timesteps = current_timesteps[timestep_indices]
            student_timesteps = current_timesteps[timestep_indices + current_skip_steps]

            # 3. Sample noise and add it to the clean images for both teacher and student unets
            # Sample noise z ~ N(0, I) that we'll add to the images
            noise = torch.randn(clean_images.shape, dtype=weight_dtype, device=clean_images.device)
            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            teacher_noisy_images = add_noise(clean_images, noise, teacher_timesteps)
            student_noisy_images = add_noise(clean_images, noise, student_timesteps)

            # 4. Calculate preconditioning and scalings for boundary conditions for the consistency model.
            teacher_rescaled_timesteps = get_noise_preconditioning(teacher_timesteps, args.noise_precond_type)
            student_rescaled_timesteps = get_noise_preconditioning(student_timesteps, args.noise_precond_type)

            c_in_teacher = get_input_preconditioning(teacher_timesteps, input_precond_type=args.input_precond_type)
            c_in_student = get_input_preconditioning(student_timesteps, input_precond_type=args.input_precond_type)

            c_skip_teacher, c_out_teacher = scalings_for_boundary_conditions(teacher_timesteps)
            c_skip_student, c_out_student = scalings_for_boundary_conditions(student_timesteps)

            c_skip_teacher, c_out_teacher, c_in_teacher = [
                append_dims(x, clean_images.ndim) for x in [c_skip_teacher, c_out_teacher, c_in_teacher]
            ]
            c_skip_student, c_out_student, c_in_student = [
                append_dims(x, clean_images.ndim) for x in [c_skip_student, c_out_student, c_in_student]
            ]

            with accelerator.accumulate(unet):
                # 5. Get the student unet denoising prediction on the student timesteps
                # Get rng state now to ensure that dropout is synced between the student and teacher models.
                dropout_state = torch.get_rng_state()
                student_model_output = unet(
                    c_in_student * student_noisy_images, student_rescaled_timesteps, class_labels=class_labels
                ).sample
                # NOTE: currently only support prediction_type == sample, so no need to convert model_output
                student_denoise_output = c_skip_student * student_noisy_images + c_out_student * student_model_output

                # 6. Get the teacher unet denoising prediction on the teacher timesteps
                with torch.no_grad(), torch.autocast("cuda", dtype=teacher_dtype):
                    torch.set_rng_state(dropout_state)
                    teacher_model_output = teacher_unet(
                        c_in_teacher * teacher_noisy_images, teacher_rescaled_timesteps, class_labels=class_labels
                    ).sample
                    # NOTE: currently only support prediction_type == sample, so no need to convert model_output
                    teacher_denoise_output = (
                        c_skip_teacher * teacher_noisy_images + c_out_teacher * teacher_model_output
                    )

                # 7. Calculate the weighted Pseudo-Huber loss
                if args.prediction_type == "sample":
                    # Note that the loss weights should be those at the (teacher) timestep indices.
                    lambda_t = _extract_into_tensor(
                        timestep_loss_weights, timestep_indices, (bsz,) + (1,) * (clean_images.ndim - 1)
                    )
                    loss = lambda_t * (
                        torch.sqrt(
                            (student_denoise_output.float() - teacher_denoise_output.float()) ** 2 + args.huber_c**2
                        )
                        - args.huber_c
                    )
                    loss = loss.mean()
                else:
                    raise ValueError(
                        f"Unsupported prediction type: {args.prediction_type}. Currently, only `sample` is supported."
                    )

                # 8. Backpropagate on the consistency training loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # 9. Update teacher_unet and ema_unet parameters using unet's parameters.
                teacher_unet.load_state_dict(unet.state_dict())
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    # 10. Recalculate quantities depending on the global step, if necessary.
                    new_discretization_steps = get_discretization_steps(
                        global_step,
                        args.max_train_steps,
                        s_0=args.discretization_s_0,
                        s_1=args.discretization_s_1,
                        constant=args.constant_discretization_steps,
                    )
                    current_skip_steps = get_skip_steps(global_step, initial_skip=args.skip_steps)
                    if current_skip_steps >= new_discretization_steps:
                        raise ValueError(
                            f"The current skip steps is {current_skip_steps}, but should be smaller than the current"
                            f" number of discretization steps {new_discretization_steps}."
                        )
                    if new_discretization_steps != current_discretization_steps:
                        (
                            noise_scheduler,
                            current_timesteps,
                            timestep_weights,
                            timestep_loss_weights,
                        ) = recalculate_num_discretization_step_values(new_discretization_steps, current_skip_steps)
                        current_discretization_steps = new_discretization_steps

                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if global_step % args.validation_steps == 0:
                        # NOTE: since we do not use EMA for the teacher model, the teacher parameters and student
                        # parameters are the same at this point in time
                        log_validation(unet, noise_scheduler, args, accelerator, weight_dtype, global_step, "teacher")
                        # teacher_unet.to(dtype=teacher_dtype)

                        if args.use_ema:
                            # Store the student unet weights and load the EMA weights.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())

                            log_validation(
                                unet,
                                noise_scheduler,
                                args,
                                accelerator,
                                weight_dtype,
                                global_step,
                                "ema_student",
                            )

                            # Restore student unet weights
                            ema_unet.restore(unet.parameters())

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if args.use_ema:
                logs["ema_decay"] = ema_unet.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        # progress_bar.close()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)
        pipeline = ConsistencyModelPipeline(unet=unet, scheduler=noise_scheduler)
        pipeline.save_pretrained(args.output_dir)

        # If using EMA, save EMA weights as well.
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

            unet.save_pretrained(os.path.join(args.output_dir, "ema_unet"))

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)