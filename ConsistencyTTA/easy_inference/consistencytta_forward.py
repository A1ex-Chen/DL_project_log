def forward(self, prompt: str, cfg_scale_input: float=3.0, cfg_scale_post:
    float=1.0, num_steps: int=1, num_samples: int=1, sr: int=16000):
    self.check_eval_mode()
    device = self.text_encoder.device
    use_cf_guidance = cfg_scale_post > 1.0
    prompt_embeds_cf, prompt_mask_cf, prompt_embeds, prompt_mask = (self.
        encode_text_classifier_free(prompt, num_samples))
    encoder_states, encoder_att_mask = (prompt_embeds_cf, prompt_mask_cf
        ) if use_cf_guidance else (prompt_embeds, prompt_mask)
    num_channels_latents = self.unet.config.in_channels
    latent_shape = len(prompt) * num_samples, num_channels_latents, 256, 16
    noise = randn_tensor(latent_shape, generator=None, device=device, dtype
        =prompt_embeds.dtype)
    self.scheduler.set_timesteps(18, device=device)
    z_N = noise * self.scheduler.init_noise_sigma

    def calc_zhat_0(z_n: Tensor, t: int):
        """ Query the consistency model to get zhat_0, which is the denoised embedding.
            Args:
                z_n (Tensor):   The noisy embedding.
                t (int):        The time step.
            Returns:
                Tensor:         The denoised embedding.
            """
        z_n_input = torch.cat([z_n] * 2) if use_cf_guidance else z_n
        z_n_input = self.scheduler.scale_model_input(z_n_input, t)
        zhat_0 = self.unet(z_n_input, t, guidance=cfg_scale_input,
            encoder_hidden_states=encoder_states, encoder_attention_mask=
            encoder_att_mask).sample
        if use_cf_guidance:
            zhat_0_uncond, zhat_0_cond = zhat_0.chunk(2)
            zhat_0 = (1 - cfg_scale_post
                ) * zhat_0_uncond + cfg_scale_post * zhat_0_cond
        return zhat_0
    zhat_0 = calc_zhat_0(z_N, self.scheduler.timesteps[0])
    self.scheduler.set_timesteps(num_steps, device=device)
    for t in self.scheduler.timesteps[1::2]:
        zhat_n = self.scheduler.add_noise(zhat_0, torch.randn_like(zhat_0), t)
        zhat_0 = calc_zhat_0(zhat_n, t)
    mel = self.vae.decode_first_stage(zhat_0.float())
    return self.vae.decode_to_waveform(mel)[:, :int(sr * 9.5)]
