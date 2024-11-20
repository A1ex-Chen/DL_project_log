# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
# limitations under the License.

import argparse
import copy
import logging
import math
import os
import shutil
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import DataLoader, Dataset, default_collate
from torchvision import transforms
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

import diffusers.optimization
from diffusers import AmusedPipeline, AmusedScheduler, EMAModel, UVit2DModel, VQModel
from diffusers.loaders import LoraLoaderMixin
from diffusers.utils import is_wandb_available


if is_wandb_available():
    import wandb

logger = get_logger(__name__, log_level="INFO")




class InstanceDataRootDataset(Dataset):
    def __init__(
        self,
        instance_data_root,
        tokenizer,
        size=512,
    ):
        self.size = size
        self.tokenizer = tokenizer
        self.instance_images_path = list(Path(instance_data_root).iterdir())

    def __len__(self):
        return len(self.instance_images_path)

    def __getitem__(self, index):
        image_path = self.instance_images_path[index % len(self.instance_images_path)]
        instance_image = Image.open(image_path)
        rv = process_image(instance_image, self.size)

        prompt = os.path.splitext(os.path.basename(image_path))[0]
        rv["prompt_input_ids"] = tokenize_prompt(self.tokenizer, prompt)[0]
        return rv


class InstanceDataImageDataset(Dataset):
    def __init__(
        self,
        instance_data_image,
        train_batch_size,
        size=512,
    ):
        self.value = process_image(Image.open(instance_data_image), size)
        self.train_batch_size = train_batch_size

    def __len__(self):
        # Needed so a full batch of the data can be returned. Otherwise will return
        # batches of size 1
        return self.train_batch_size

    def __getitem__(self, index):
        return self.value


class HuggingFaceDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        tokenizer,
        image_key,
        prompt_key,
        prompt_prefix=None,
        size=512,
    ):
        self.size = size
        self.image_key = image_key
        self.prompt_key = prompt_key
        self.tokenizer = tokenizer
        self.hf_dataset = hf_dataset
        self.prompt_prefix = prompt_prefix

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, index):
        item = self.hf_dataset[index]

        rv = process_image(item[self.image_key], self.size)

        prompt = item[self.prompt_key]

        if self.prompt_prefix is not None:
            prompt = self.prompt_prefix + prompt

        rv["prompt_input_ids"] = tokenize_prompt(self.tokenizer, prompt)[0]

        return rv














class InstanceDataImageDataset(Dataset):




class HuggingFaceDataset(Dataset):




def process_image(image, size):
    image = exif_transpose(image)

    if not image.mode == "RGB":
        image = image.convert("RGB")

    orig_height = image.height
    orig_width = image.width

    image = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)(image)

    c_top, c_left, _, _ = transforms.RandomCrop.get_params(image, output_size=(size, size))
    image = transforms.functional.crop(image, c_top, c_left, size, size)

    image = transforms.ToTensor()(image)

    micro_conds = torch.tensor(
        [orig_width, orig_height, c_top, c_left, 6.0],
    )

    return {"image": image, "micro_conds": micro_conds}


def tokenize_prompt(tokenizer, prompt):
    return tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=77,
        return_tensors="pt",
    ).input_ids


def encode_prompt(text_encoder, input_ids):
    outputs = text_encoder(input_ids, return_dict=True, output_hidden_states=True)
    encoder_hidden_states = outputs.hidden_states[-2]
    cond_embeds = outputs[0]
    return encoder_hidden_states, cond_embeds


def main(args):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_main_process:
        accelerator.init_trackers("amused", config=vars(copy.deepcopy(args)))

    if args.seed is not None:
        set_seed(args.seed)

    # TODO - will have to fix loading if training text encoder
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, variant=args.variant
    )
    vq_model = VQModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vqvae", revision=args.revision, variant=args.variant
    )

    if args.train_text_encoder:
        if args.text_encoder_use_lora:
            lora_config = LoraConfig(
                r=args.text_encoder_lora_r,
                lora_alpha=args.text_encoder_lora_alpha,
                target_modules=args.text_encoder_lora_target_modules,
            )
            text_encoder.add_adapter(lora_config)
        text_encoder.train()
        text_encoder.requires_grad_(True)
    else:
        text_encoder.eval()
        text_encoder.requires_grad_(False)

    vq_model.requires_grad_(False)

    model = UVit2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
    )

    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
        )
        model.add_adapter(lora_config)

    model.train()

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if args.use_ema:
        ema = EMAModel(
            model.parameters(),
            decay=args.ema_decay,
            update_after_step=args.ema_update_after_step,
            model_cls=UVit2DModel,
            model_config=model.config,
        )



    accelerator.register_load_state_pre_hook(load_model_hook)
    accelerator.register_save_state_pre_hook(save_model_hook)

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.adam_weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    if args.train_text_encoder:
        optimizer_grouped_parameters.append(
            {"params": text_encoder.parameters(), "weight_decay": args.adam_weight_decay}
        )

    optimizer = optimizer_cls(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    logger.info("Creating dataloaders and lr_scheduler")

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    if args.instance_data_dir is not None:
        dataset = InstanceDataRootDataset(
            instance_data_root=args.instance_data_dir,
            tokenizer=tokenizer,
            size=args.resolution,
        )
    elif args.instance_data_image is not None:
        dataset = InstanceDataImageDataset(
            instance_data_image=args.instance_data_image,
            train_batch_size=args.train_batch_size,
            size=args.resolution,
        )
    elif args.instance_data_dataset is not None:
        dataset = HuggingFaceDataset(
            hf_dataset=load_dataset(args.instance_data_dataset, split="train"),
            tokenizer=tokenizer,
            image_key=args.image_key,
            prompt_key=args.prompt_key,
            prompt_prefix=args.prompt_prefix,
            size=args.resolution,
        )
    else:
        assert False

    train_dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=default_collate,
    )
    train_dataloader.num_batches = len(train_dataloader)

    lr_scheduler = diffusers.optimization.get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
    )

    logger.info("Preparing model, optimizer and dataloaders")

    if args.train_text_encoder:
        model, optimizer, lr_scheduler, train_dataloader, text_encoder = accelerator.prepare(
            model, optimizer, lr_scheduler, train_dataloader, text_encoder
        )
    else:
        model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
            model, optimizer, lr_scheduler, train_dataloader
        )

    train_dataloader.num_batches = len(train_dataloader)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if not args.train_text_encoder:
        text_encoder.to(device=accelerator.device, dtype=weight_dtype)

    vq_model.to(device=accelerator.device)

    if args.use_ema:
        ema.to(accelerator.device)

    with nullcontext() if args.train_text_encoder else torch.no_grad():
        empty_embeds, empty_clip_embeds = encode_prompt(
            text_encoder, tokenize_prompt(tokenizer, "").to(text_encoder.device, non_blocking=True)
        )

        # There is a single image, we can just pre-encode the single prompt
        if args.instance_data_image is not None:
            prompt = os.path.splitext(os.path.basename(args.instance_data_image))[0]
            encoder_hidden_states, cond_embeds = encode_prompt(
                text_encoder, tokenize_prompt(tokenizer, prompt).to(text_encoder.device, non_blocking=True)
            )
            encoder_hidden_states = encoder_hidden_states.repeat(args.train_batch_size, 1, 1)
            cond_embeds = cond_embeds.repeat(args.train_batch_size, 1)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(train_dataloader.num_batches / args.gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs.
    # Note: We are not doing epoch based training here, but just using this for book keeping and being able to
    # reuse the same training loop with other datasets/loaders.
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {args.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = { args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint:
        if resume_from_checkpoint == "latest":
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            if len(dirs) > 0:
                resume_from_checkpoint = os.path.join(args.output_dir, dirs[-1])
            else:
                resume_from_checkpoint = None

        if resume_from_checkpoint is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
        else:
            accelerator.print(f"Resuming from checkpoint {resume_from_checkpoint}")

    if resume_from_checkpoint is None:
        global_step = 0
        first_epoch = 0
    else:
        accelerator.load_state(resume_from_checkpoint)
        global_step = int(os.path.basename(resume_from_checkpoint).split("-")[1])
        first_epoch = global_step // num_update_steps_per_epoch

    # As stated above, we are not doing epoch based training here, but just using this for book keeping and being able to
    # reuse the same training loop with other datasets/loaders.
    for epoch in range(first_epoch, num_train_epochs):
        for batch in train_dataloader:
            with torch.no_grad():
                micro_conds = batch["micro_conds"].to(accelerator.device, non_blocking=True)
                pixel_values = batch["image"].to(accelerator.device, non_blocking=True)

                batch_size = pixel_values.shape[0]

                split_batch_size = args.split_vae_encode if args.split_vae_encode is not None else batch_size
                num_splits = math.ceil(batch_size / split_batch_size)
                image_tokens = []
                for i in range(num_splits):
                    start_idx = i * split_batch_size
                    end_idx = min((i + 1) * split_batch_size, batch_size)
                    bs = pixel_values.shape[0]
                    image_tokens.append(
                        vq_model.quantize(vq_model.encode(pixel_values[start_idx:end_idx]).latents)[2][2].reshape(
                            bs, -1
                        )
                    )
                image_tokens = torch.cat(image_tokens, dim=0)

                batch_size, seq_len = image_tokens.shape

                timesteps = torch.rand(batch_size, device=image_tokens.device)
                mask_prob = torch.cos(timesteps * math.pi * 0.5)
                mask_prob = mask_prob.clip(args.min_masking_rate)

                num_token_masked = (seq_len * mask_prob).round().clamp(min=1)
                batch_randperm = torch.rand(batch_size, seq_len, device=image_tokens.device).argsort(dim=-1)
                mask = batch_randperm < num_token_masked.unsqueeze(-1)

                mask_id = accelerator.unwrap_model(model).config.vocab_size - 1
                input_ids = torch.where(mask, mask_id, image_tokens)
                labels = torch.where(mask, image_tokens, -100)

                if args.cond_dropout_prob > 0.0:
                    assert encoder_hidden_states is not None

                    batch_size = encoder_hidden_states.shape[0]

                    mask = (
                        torch.zeros((batch_size, 1, 1), device=encoder_hidden_states.device).float().uniform_(0, 1)
                        < args.cond_dropout_prob
                    )

                    empty_embeds_ = empty_embeds.expand(batch_size, -1, -1)
                    encoder_hidden_states = torch.where(
                        (encoder_hidden_states * mask).bool(), encoder_hidden_states, empty_embeds_
                    )

                    empty_clip_embeds_ = empty_clip_embeds.expand(batch_size, -1)
                    cond_embeds = torch.where((cond_embeds * mask.squeeze(-1)).bool(), cond_embeds, empty_clip_embeds_)

                bs = input_ids.shape[0]
                vae_scale_factor = 2 ** (len(vq_model.config.block_out_channels) - 1)
                resolution = args.resolution // vae_scale_factor
                input_ids = input_ids.reshape(bs, resolution, resolution)

            if "prompt_input_ids" in batch:
                with nullcontext() if args.train_text_encoder else torch.no_grad():
                    encoder_hidden_states, cond_embeds = encode_prompt(
                        text_encoder, batch["prompt_input_ids"].to(accelerator.device, non_blocking=True)
                    )

            # Train Step
            with accelerator.accumulate(model):
                codebook_size = accelerator.unwrap_model(model).config.codebook_size

                logits = (
                    model(
                        input_ids=input_ids,
                        encoder_hidden_states=encoder_hidden_states,
                        micro_conds=micro_conds,
                        pooled_text_emb=cond_embeds,
                    )
                    .reshape(bs, codebook_size, -1)
                    .permute(0, 2, 1)
                    .reshape(-1, codebook_size)
                )

                loss = F.cross_entropy(
                    logits,
                    labels.view(-1),
                    ignore_index=-100,
                    reduction="mean",
                )

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                avg_masking_rate = accelerator.gather(mask_prob.repeat(args.train_batch_size)).mean()

                accelerator.backward(loss)

                if args.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()

                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema.step(model.parameters())

                if (global_step + 1) % args.logging_steps == 0:
                    logs = {
                        "step_loss": avg_loss.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "avg_masking_rate": avg_masking_rate.item(),
                    }
                    accelerator.log(logs, step=global_step + 1)

                    logger.info(
                        f"Step: {global_step + 1} "
                        f"Loss: {avg_loss.item():0.4f} "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.6f}"
                    )

                if (global_step + 1) % args.checkpointing_steps == 0:
                    save_checkpoint(args, accelerator, global_step + 1)

                if (global_step + 1) % args.validation_steps == 0 and accelerator.is_main_process:
                    if args.use_ema:
                        ema.store(model.parameters())
                        ema.copy_to(model.parameters())

                    with torch.no_grad():
                        logger.info("Generating images...")

                        model.eval()

                        if args.train_text_encoder:
                            text_encoder.eval()

                        scheduler = AmusedScheduler.from_pretrained(
                            args.pretrained_model_name_or_path,
                            subfolder="scheduler",
                            revision=args.revision,
                            variant=args.variant,
                        )

                        pipe = AmusedPipeline(
                            transformer=accelerator.unwrap_model(model),
                            tokenizer=tokenizer,
                            text_encoder=text_encoder,
                            vqvae=vq_model,
                            scheduler=scheduler,
                        )

                        pil_images = pipe(prompt=args.validation_prompts).images
                        wandb_images = [
                            wandb.Image(image, caption=args.validation_prompts[i])
                            for i, image in enumerate(pil_images)
                        ]

                        wandb.log({"generated_images": wandb_images}, step=global_step + 1)

                        model.train()

                        if args.train_text_encoder:
                            text_encoder.train()

                    if args.use_ema:
                        ema.restore(model.parameters())

                global_step += 1

            # Stop training if max steps is reached
            if global_step >= args.max_train_steps:
                break
        # End for

    accelerator.wait_for_everyone()

    # Evaluate and save checkpoint at the end of training
    save_checkpoint(args, accelerator, global_step)

    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        if args.use_ema:
            ema.copy_to(model.parameters())
        model.save_pretrained(args.output_dir)

    accelerator.end_training()


def save_checkpoint(args, accelerator, global_step):
    output_dir = args.output_dir

    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if accelerator.is_main_process and args.checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
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
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = Path(output_dir) / f"checkpoint-{global_step}"
    accelerator.save_state(save_path)
    logger.info(f"Saved state to {save_path}")


if __name__ == "__main__":
    main(parse_args())