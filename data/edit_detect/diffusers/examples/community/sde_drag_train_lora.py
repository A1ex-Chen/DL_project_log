def train_lora(self, prompt, image, lora_step=100, lora_rank=16, generator=None
    ):
    accelerator = Accelerator(gradient_accumulation_steps=1,
        mixed_precision='fp16')
    self.vae.requires_grad_(False)
    self.text_encoder.requires_grad_(False)
    self.unet.requires_grad_(False)
    unet_lora_attn_procs = {}
    for name, attn_processor in self.unet.attn_processors.items():
        cross_attention_dim = None if name.endswith('attn1.processor'
            ) else self.unet.config.cross_attention_dim
        if name.startswith('mid_block'):
            hidden_size = self.unet.config.block_out_channels[-1]
        elif name.startswith('up_blocks'):
            block_id = int(name[len('up_blocks.')])
            hidden_size = list(reversed(self.unet.config.block_out_channels))[
                block_id]
        elif name.startswith('down_blocks'):
            block_id = int(name[len('down_blocks.')])
            hidden_size = self.unet.config.block_out_channels[block_id]
        else:
            raise NotImplementedError(
                'name must start with up_blocks, mid_blocks, or down_blocks')
        if isinstance(attn_processor, (AttnAddedKVProcessor,
            SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_0)):
            lora_attn_processor_class = LoRAAttnAddedKVProcessor
        else:
            lora_attn_processor_class = LoRAAttnProcessor2_0 if hasattr(torch
                .nn.functional, 'scaled_dot_product_attention'
                ) else LoRAAttnProcessor
        unet_lora_attn_procs[name] = lora_attn_processor_class(hidden_size=
            hidden_size, cross_attention_dim=cross_attention_dim, rank=
            lora_rank)
    self.unet.set_attn_processor(unet_lora_attn_procs)
    unet_lora_layers = AttnProcsLayers(self.unet.attn_processors)
    params_to_optimize = unet_lora_layers.parameters()
    optimizer = torch.optim.AdamW(params_to_optimize, lr=0.0002, betas=(0.9,
        0.999), weight_decay=0.01, eps=1e-08)
    lr_scheduler = get_scheduler('constant', optimizer=optimizer,
        num_warmup_steps=0, num_training_steps=lora_step, num_cycles=1,
        power=1.0)
    unet_lora_layers = accelerator.prepare_model(unet_lora_layers)
    optimizer = accelerator.prepare_optimizer(optimizer)
    lr_scheduler = accelerator.prepare_scheduler(lr_scheduler)
    with torch.no_grad():
        text_inputs = self._tokenize_prompt(prompt, tokenizer_max_length=None)
        text_embedding = self._encode_prompt(text_inputs.input_ids,
            text_inputs.attention_mask, text_encoder_use_attention_mask=False)
    image_transforms = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])])
    image = image_transforms(image).to(self.device, dtype=self.vae.dtype)
    image = image.unsqueeze(dim=0)
    latents_dist = self.vae.encode(image).latent_dist
    for _ in tqdm(range(lora_step), desc='Train LoRA'):
        self.unet.train()
        model_input = latents_dist.sample() * self.vae.config.scaling_factor
        noise = torch.randn(model_input.size(), dtype=model_input.dtype,
            layout=model_input.layout, device=model_input.device, generator
            =generator)
        bsz, channels, height, width = model_input.shape
        timesteps = torch.randint(0, self.scheduler.config.
            num_train_timesteps, (bsz,), device=model_input.device,
            generator=generator)
        timesteps = timesteps.long()
        noisy_model_input = self.scheduler.add_noise(model_input, noise,
            timesteps)
        model_pred = self.unet(noisy_model_input, timesteps, text_embedding
            ).sample
        if self.scheduler.config.prediction_type == 'epsilon':
            target = noise
        elif self.scheduler.config.prediction_type == 'v_prediction':
            target = self.scheduler.get_velocity(model_input, noise, timesteps)
        else:
            raise ValueError(
                f'Unknown prediction type {self.scheduler.config.prediction_type}'
                )
        loss = torch.nn.functional.mse_loss(model_pred.float(), target.
            float(), reduction='mean')
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    with tempfile.TemporaryDirectory() as save_lora_dir:
        LoraLoaderMixin.save_lora_weights(save_directory=save_lora_dir,
            unet_lora_layers=unet_lora_layers, text_encoder_lora_layers=None)
        self.unet.load_attn_procs(save_lora_dir)
