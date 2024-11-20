@torch.no_grad()
def inference(self, prompt, inference_scheduler, guidance_scale_input=3,
    guidance_scale_post=1, num_steps=20, use_edm=False, num_samples=1,
    use_ema=True, query_teacher=False, **kwargs):
    self.check_eval_mode()
    device = self.text_encoder.device
    batch_size = len(prompt) * num_samples
    use_cf_guidance = guidance_scale_post > 1.0
    prompt_embeds_cf, prompt_mask_cf, prompt_embeds, prompt_mask = (self.
        encode_text_classifier_free(prompt, num_samples))
    encoder_states_stu, encoder_att_mask_stu = (prompt_embeds_cf,
        prompt_mask_cf) if use_cf_guidance else (prompt_embeds, prompt_mask)
    encoder_states_tea, encoder_att_mask_tea = (prompt_embeds_cf,
        prompt_mask_cf) if self.use_teacher_cf_guidance else (prompt_embeds,
        prompt_mask)
    inference_scheduler.set_timesteps(num_steps, device=device)
    timesteps = inference_scheduler.timesteps
    if query_teacher:
        inference_scheduler_tea = deepcopy(inference_scheduler)
    latent_shape = batch_size, self.student_unet.config.in_channels, 256, 16
    zhat_N = randn_tensor(latent_shape, generator=None, device=device,
        dtype=prompt_embeds.dtype) * inference_scheduler.init_noise_sigma
    zhat_n_stu, zhat_n_tea = zhat_N, zhat_N
    for t in timesteps:
        zhat_n_input_stu = torch.cat([zhat_n_stu] * 2
            ) if use_cf_guidance else zhat_n_stu
        zhat_n_input_stu = inference_scheduler.scale_model_input(
            zhat_n_input_stu, t)
        unet = self.student_ema_unet if use_ema else self.student_unet
        noise_pred_stu = unet(zhat_n_input_stu, t, guidance=
            guidance_scale_input, encoder_hidden_states=encoder_states_stu,
            encoder_attention_mask=encoder_att_mask_stu).sample
        if use_cf_guidance:
            noise_pred_uncond_stu, noise_pred_cond_stu = noise_pred_stu.chunk(2
                )
            noise_pred_stu = noise_pred_uncond_stu + guidance_scale_post * (
                noise_pred_cond_stu - noise_pred_uncond_stu)
        zhat_n_stu = inference_scheduler.step(noise_pred_stu, t, zhat_n_stu
            ).prev_sample
        if query_teacher:
            zhat_n_scaled_tea = inference_scheduler.scale_model_input(
                zhat_n_tea, t)
            noise_pred_tea = self._query_teacher(zhat_n_scaled_tea, t,
                encoder_states_tea, encoder_att_mask_tea, guidance_scale_input)
            zhat_n_tea = inference_scheduler_tea.step(noise_pred_tea, t,
                zhat_n_tea).prev_sample
    if query_teacher:
        logger.info(
            f'Loss w.r.t. teacher: {F.mse_loss(zhat_n_tea, zhat_n_stu):.3f}.')
    return zhat_n_stu
