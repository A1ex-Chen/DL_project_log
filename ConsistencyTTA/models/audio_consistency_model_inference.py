@torch.no_grad()
def inference(self, prompt, inference_scheduler, guidance_scale_input=3,
    guidance_scale_post=1, num_steps=20, use_edm=False, num_samples=1,
    use_ema=True, query_teacher=False, num_teacher_steps=18, return_all=False):

    def calc_zhat_0(z_n, t, prompt_embeds, prompt_mask,
        guidance_scale_input, guidance_scale_post):
        use_cf_guidance = guidance_scale_post > 1.0
        z_n_input = torch.cat([z_n] * 2) if use_cf_guidance else z_n
        z_n_input = inference_scheduler.scale_model_input(z_n_input, t)
        unet = self.student_ema_unet if use_ema else self.student_target_unet
        zhat_0 = unet(z_n_input, t, guidance=guidance_scale_input,
            encoder_hidden_states=prompt_embeds, encoder_attention_mask=
            prompt_mask).sample
        if use_cf_guidance:
            zhat_0_uncond, zhat_0_cond = zhat_0.chunk(2)
            zhat_0 = (1 - guidance_scale_post
                ) * zhat_0_uncond + guidance_scale_post * zhat_0_cond
        return zhat_0
    self.check_eval_mode()
    device = self.text_encoder.device
    batch_size = len(prompt) * num_samples
    use_cf_guidance = guidance_scale_post > 1.0
    t_start_embed = time()
    prompt_embeds_cf, prompt_mask_cf, prompt_embeds, prompt_mask = (self.
        encode_text_classifier_free(prompt, num_samples))
    encoder_states_stu, encoder_att_mask_stu = (prompt_embeds_cf,
        prompt_mask_cf) if use_cf_guidance else (prompt_embeds, prompt_mask)
    encoder_states_tea, encoder_att_mask_tea = (prompt_embeds_cf,
        prompt_mask_cf) if self.use_teacher_cf_guidance else (prompt_embeds,
        prompt_mask)
    num_channels_latents = self.student_target_unet.config.in_channels
    latent_shape = batch_size, num_channels_latents, 256, 16
    noise = randn_tensor(latent_shape, generator=None, device=device, dtype
        =prompt_embeds.dtype)
    time_embed = time() - t_start_embed
    t_start_stu = time()
    inference_scheduler.set_timesteps(18, device=device)
    z_N_stu = noise * inference_scheduler.init_noise_sigma
    zhat_0_stu = calc_zhat_0(z_N_stu, inference_scheduler.timesteps[0],
        encoder_states_stu, encoder_att_mask_stu, guidance_scale_input,
        guidance_scale_post)
    inference_scheduler.set_timesteps(num_steps, device=device)
    order = 2 if self.use_edm else 1
    for t in inference_scheduler.timesteps[1::order]:
        zhat_n_stu = inference_scheduler.add_noise(zhat_0_stu, torch.
            randn_like(zhat_0_stu), t)
        zhat_0_stu = calc_zhat_0(zhat_n_stu, t, encoder_states_stu,
            encoder_att_mask_stu, guidance_scale_input, guidance_scale_post)
    time_stu = time() - t_start_stu
    if return_all:
        print('Distilled model generation completed!')
    if query_teacher:
        t_start_tea = time()
        inference_scheduler.set_timesteps(num_teacher_steps, device=device)
        zhat_n_tea = noise * inference_scheduler.init_noise_sigma
        for t in inference_scheduler.timesteps:
            zhat_n_input = inference_scheduler.scale_model_input(zhat_n_tea, t)
            noise_pred = self._query_teacher(zhat_n_input, t,
                encoder_states_tea, encoder_att_mask_tea, guidance_scale_input)
            zhat_n_tea = inference_scheduler.step(noise_pred, t, zhat_n_tea
                ).prev_sample
        if self.use_edm:
            inference_scheduler.prev_derivative = None
            inference_scheduler.dt = None
            inference_scheduler.sample = None
        time_tea = time() - t_start_tea
        if return_all:
            print('Diffusion model generation completed!')
    else:
        zhat_n_tea, time_tea = None, None
    if return_all:
        if time_tea is not None:
            time_tea += time_embed
        return zhat_0_stu, zhat_n_tea, time_stu + time_embed, time_tea
    else:
        return zhat_0_stu
