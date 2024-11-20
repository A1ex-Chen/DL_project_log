def forward(self, z_0, prompt, **kwargs):
    """ z_0:    Ground-truth latent variables.
            prompt: Text prompt for the generation.
        """

    def get_loss(model_pred, target, timesteps):
        if self.snr_gamma is None:
            return F.mse_loss(model_pred.float(), target.float(), reduction
                ='mean')
        else:
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor(timesteps)
            assert len(timesteps.shape) < 2
            timesteps = timesteps.reshape(-1)
            snr = self.compute_snr(timesteps).reshape(-1)
            truncated_snr = torch.clamp(snr, max=self.snr_gamma)
            if self.noise_scheduler.config.prediction_type == 'v_prediction':
                mse_loss_weights = truncated_snr / (snr + 1)
            elif self.noise_scheduler.config.prediction_type == 'epsilon':
                mse_loss_weights = truncated_snr / snr
            else:
                raise ValueError('Unknown prediction type.')
            loss = F.mse_loss(model_pred.float(), target.float(), reduction
                ='none')
            instance_loss = loss.mean(dim=list(range(1, len(loss.shape))))
            return (instance_loss * mse_loss_weights.to(loss.device)).mean()

    def get_random_timestep(batch_size):
        device = self.text_encoder.device
        avail_timesteps = self.noise_scheduler.timesteps.to(device)
        time_inds = torch.randint(0, len(avail_timesteps), (batch_size,))
        t_n = avail_timesteps[time_inds.to(device)]
        return t_n
    self.check_eval_mode()
    prompt_embeds_cf, prompt_mask_cf, prompt_embeds, prompt_mask = (self.
        get_prompt_embeds(prompt, self.use_teacher_cf_guidance,
        num_samples_per_prompt=1))
    t_n = get_random_timestep(z_0.shape[0])
    gaussian_noise = torch.randn_like(z_0)
    z_noisy = self.noise_scheduler.add_noise(z_0, gaussian_noise, t_n)
    z_gaussian = gaussian_noise * self.noise_scheduler.init_noise_sigma
    last_mask = (t_n == self.noise_scheduler.timesteps.max()).reshape(-1, 1,
        1, 1)
    z_n = torch.where(last_mask, z_gaussian.to(z_0.device), z_noisy)
    z_n_scaled = self.noise_scheduler.scale_model_input(z_n, t_n)
    if self.teacher_guidance_scale == -1:
        guidance_scale = torch.rand(z_0.shape[0]
            ) * self.max_rand_guidance_scale
        guidance_scale = guidance_scale.to(z_0.device)
    else:
        guidance_scale = None
    with torch.no_grad():
        noise_pred_teacher = self._query_teacher(z_n_scaled, t_n,
            prompt_embeds_cf, prompt_mask_cf, guidance_scale)
    noise_pred_student = self.student_unet(z_n_scaled, t_n, guidance=
        guidance_scale, encoder_hidden_states=prompt_embeds,
        encoder_attention_mask=prompt_mask).sample
    return get_loss(noise_pred_student, noise_pred_teacher, t_n)
