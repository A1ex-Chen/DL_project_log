@torch.no_grad()
def __call__(self, canvas_height: int, canvas_width: int, regions: List[
    DiffusionRegion], num_inference_steps: Optional[int]=50, seed: Optional
    [int]=12345, reroll_regions: Optional[List[RerollRegion]]=None, cpu_vae:
    Optional[bool]=False, decode_steps: Optional[bool]=False):
    if reroll_regions is None:
        reroll_regions = []
    batch_size = 1
    if decode_steps:
        steps_images = []
    self.scheduler.set_timesteps(num_inference_steps, device=self.device)
    text2image_regions = [region for region in regions if isinstance(region,
        Text2ImageRegion)]
    image2image_regions = [region for region in regions if isinstance(
        region, Image2ImageRegion)]
    for region in text2image_regions:
        region.tokenize_prompt(self.tokenizer)
        region.encode_prompt(self.text_encoder, self.device)
    latents_shape = (batch_size, self.unet.config.in_channels, 
        canvas_height // 8, canvas_width // 8)
    generator = torch.Generator(self.device).manual_seed(seed)
    init_noise = torch.randn(latents_shape, generator=generator, device=
        self.device)
    for region in reroll_regions:
        if region.reroll_mode == RerollModes.RESET.value:
            region_shape = (latents_shape[0], latents_shape[1], region.
                latent_row_end - region.latent_row_init, region.
                latent_col_end - region.latent_col_init)
            init_noise[:, :, region.latent_row_init:region.latent_row_end,
                region.latent_col_init:region.latent_col_end] = torch.randn(
                region_shape, generator=region.get_region_generator(self.
                device), device=self.device)
    all_eps_rerolls = regions + [r for r in reroll_regions if r.reroll_mode ==
        RerollModes.EPSILON.value]
    for region in all_eps_rerolls:
        if region.noise_eps > 0:
            region_noise = init_noise[:, :, region.latent_row_init:region.
                latent_row_end, region.latent_col_init:region.latent_col_end]
            eps_noise = torch.randn(region_noise.shape, generator=region.
                get_region_generator(self.device), device=self.device
                ) * region.noise_eps
            init_noise[:, :, region.latent_row_init:region.latent_row_end,
                region.latent_col_init:region.latent_col_end] += eps_noise
    latents = init_noise * self.scheduler.init_noise_sigma
    for region in text2image_regions:
        max_length = region.tokenized_prompt.input_ids.shape[-1]
        uncond_input = self.tokenizer([''] * batch_size, padding=
            'max_length', max_length=max_length, return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(
            self.device))[0]
        region.encoded_prompt = torch.cat([uncond_embeddings, region.
            encoded_prompt])
    for region in image2image_regions:
        region.encode_reference_image(self.vae, device=self.device,
            generator=generator)
    mask_builder = MaskWeightsBuilder(latent_space_dim=self.unet.config.
        in_channels, nbatch=batch_size)
    mask_weights = [mask_builder.compute_mask_weights(region).to(self.
        device) for region in text2image_regions]
    for i, t in tqdm(enumerate(self.scheduler.timesteps)):
        noise_preds_regions = []
        for region in text2image_regions:
            region_latents = latents[:, :, region.latent_row_init:region.
                latent_row_end, region.latent_col_init:region.latent_col_end]
            latent_model_input = torch.cat([region_latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)
            noise_pred = self.unet(latent_model_input, t,
                encoder_hidden_states=region.encoded_prompt)['sample']
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_region = noise_pred_uncond + region.guidance_scale * (
                noise_pred_text - noise_pred_uncond)
            noise_preds_regions.append(noise_pred_region)
        noise_pred = torch.zeros(latents.shape, device=self.device)
        contributors = torch.zeros(latents.shape, device=self.device)
        for region, noise_pred_region, mask_weights_region in zip(
            text2image_regions, noise_preds_regions, mask_weights):
            noise_pred[:, :, region.latent_row_init:region.latent_row_end,
                region.latent_col_init:region.latent_col_end
                ] += noise_pred_region * mask_weights_region
            contributors[:, :, region.latent_row_init:region.latent_row_end,
                region.latent_col_init:region.latent_col_end
                ] += mask_weights_region
        noise_pred /= contributors
        noise_pred = torch.nan_to_num(noise_pred)
        latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        for region in image2image_regions:
            influence_step = self.get_latest_timestep_img2img(
                num_inference_steps, region.strength)
            if t > influence_step:
                timestep = t.repeat(batch_size)
                region_init_noise = init_noise[:, :, region.latent_row_init
                    :region.latent_row_end, region.latent_col_init:region.
                    latent_col_end]
                region_latents = self.scheduler.add_noise(region.
                    reference_latents, region_init_noise, timestep)
                latents[:, :, region.latent_row_init:region.latent_row_end,
                    region.latent_col_init:region.latent_col_end
                    ] = region_latents
        if decode_steps:
            steps_images.append(self.decode_latents(latents, cpu_vae))
    image = self.decode_latents(latents, cpu_vae)
    output = {'images': image}
    if decode_steps:
        output = {**output, 'steps_images': steps_images}
    return output
