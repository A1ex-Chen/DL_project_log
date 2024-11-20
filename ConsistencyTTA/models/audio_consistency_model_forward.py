def forward(self, z_0, gt_wav, prompt, validation_mode=False, run_teacher=
    True, **kwargs):
    """
        z_0:                Ground-truth latent variables.
        prompt:             Text prompt for the generation.
        validation_mode:    If 0 or False, operate in training mode and sample a random
                                timestep. If >0, operate in validation model, and then it
                                specifies the index of the discrite time step.
        run_teacher:        If True, run the teacher all the way to t=0 for validation
                                loss calculation. Otherwise, only query the teacher once.
        """

    def get_loss(model_pred, target, gt_wav, prompt, timesteps, t_indices):
        if self.snr_gamma is None:
            return self.loss(model_pred, target, gt_wav, prompt).mean()
        else:
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor(timesteps)
            assert len(timesteps.shape) < 2
            timesteps = timesteps.reshape(-1)
            snr = self.compute_snr(timesteps, t_indices).reshape(-1)
            mse_loss_weights = torch.clamp(snr, max=self.snr_gamma)
            instance_loss = self.loss(model_pred, target, gt_wav, prompt)
            return (instance_loss * mse_loss_weights.to(instance_loss.device)
                ).mean()

    def get_random_timestep(batch_size, validation_mode):
        device = self.text_encoder.device
        avail_timesteps = self.noise_scheduler.timesteps.to(device)
        order = 2 if self.use_edm else 1
        if validation_mode != 0:
            time_ind = len(avail_timesteps) - 1 - int(validation_mode * order)
            assert time_ind >= 0
            time_inds = time_ind * torch.ones((batch_size,), dtype=torch.
                int32, device=device)
        else:
            time_inds = torch.randint(0, (len(avail_timesteps) - 1) //
                order, (batch_size,), device=device) * order
        t_nplus1 = avail_timesteps[time_inds]
        t_n = avail_timesteps[time_inds + order]
        return t_nplus1, t_n, time_inds, time_inds + order
    self.check_eval_mode()
    assert validation_mode >= 0
    prompt_embeds_cf, prompt_mask_cf, prompt_embeds, prompt_mask = (self.
        get_prompt_embeds(prompt, self.use_teacher_cf_guidance,
        num_samples_per_prompt=1))
    if self.uncondition:
        raise NotImplementedError
    t_nplus1, t_n, t_ind_nplus1, t_ind_n = get_random_timestep(z_0.shape[0],
        validation_mode)
    gaussian_noise = torch.randn_like(z_0)
    z_noisy = self.noise_scheduler.add_noise(z_0, gaussian_noise, t_nplus1)
    z_gaussian = gaussian_noise * self.noise_scheduler.init_noise_sigma
    last_step = self.noise_scheduler.timesteps.max()
    last_mask = (t_nplus1 == last_step).reshape(-1, 1, 1, 1)
    z_nplus1 = torch.where(last_mask, z_gaussian.to(z_0.device), z_noisy)
    z_nplus1_scaled = self.noise_scheduler.scale_model_input(z_nplus1, t_nplus1
        )
    if self.use_edm:
        assert self.noise_scheduler.state_in_first_order
    if self.teacher_guidance_scale == -1:
        guidance_scale = torch.rand(z_0.shape[0]
            ) * self.max_rand_guidance_scale
        guidance_scale = guidance_scale.to(z_0.device)
    else:
        guidance_scale = None
    with torch.no_grad():
        noise_pred_nplus1 = self._query_teacher(z_nplus1_scaled, t_nplus1,
            prompt_embeds_cf, prompt_mask_cf, guidance_scale)
        zhat_n = self.noise_scheduler.step(noise_pred_nplus1, t_nplus1,
            z_nplus1).prev_sample
        zhat_n_scaled = self.noise_scheduler.scale_model_input(zhat_n, t_n)
        assert not zhat_n_scaled.isnan().any(
            ), f'zhat_n is NaN at t={t_nplus1}'
        if self.use_edm:
            noise_pred_n = self._query_teacher(zhat_n_scaled, t_n,
                prompt_embeds_cf, prompt_mask_cf, guidance_scale)
            zhat_n = self.noise_scheduler.step(noise_pred_n, t_n, zhat_n
                ).prev_sample
            zhat_n_scaled = self.noise_scheduler.scale_model_input(zhat_n, t_n)
            assert not zhat_n_scaled.isnan().any(
                ), f'zhat_n is NaN at t={t_nplus1}'
            assert self.noise_scheduler.state_in_first_order
    if validation_mode != 0:
        with torch.no_grad():
            zhat_0_from_nplus1 = self.student_target_unet(z_nplus1_scaled,
                t_nplus1, guidance=guidance_scale, encoder_hidden_states=
                prompt_embeds, encoder_attention_mask=prompt_mask).sample
            zhat_0_from_n = self.student_target_unet(zhat_n_scaled, t_n,
                guidance=guidance_scale, encoder_hidden_states=
                prompt_embeds, encoder_attention_mask=prompt_mask).sample
            if run_teacher:
                device = self.text_encoder.device
                avail_timesteps = self.noise_scheduler.timesteps.to(device)
                for t in avail_timesteps[t_ind_n[0]:]:
                    zhat_n_scaled_tea = self.noise_scheduler.scale_model_input(
                        zhat_n, t)
                    noise_pred_n = self._query_teacher(zhat_n_scaled_tea, t,
                        prompt_embeds_cf, prompt_mask_cf, guidance_scale)
                    zhat_n = self.noise_scheduler.step(noise_pred_n, t, zhat_n)
                    zhat_n = zhat_n.prev_sample
                    assert not zhat_n.isnan().any()
                if self.use_edm:
                    self.noise_scheduler.prev_derivative = None
                    self.noise_scheduler.dt = None
                    self.noise_scheduler.sample = None
        t_nplus1 = avail_timesteps[t_ind_nplus1[0]]
        loss_w_gt = F.mse_loss(zhat_0_from_nplus1, z_0)
        loss_w_teacher = F.mse_loss(zhat_0_from_nplus1, zhat_n)
        loss_consis = get_loss(zhat_0_from_nplus1, zhat_0_from_n, gt_wav,
            prompt, t_nplus1, t_ind_nplus1[0])
        loss_teacher = F.mse_loss(zhat_n, z_0)
        return loss_w_gt, loss_w_teacher, loss_consis, loss_teacher
    else:
        with torch.no_grad():
            zhat_0_from_n = self.student_target_unet(zhat_n_scaled, t_n,
                guidance=guidance_scale, encoder_hidden_states=
                prompt_embeds, encoder_attention_mask=prompt_mask
                ).sample.detach()
            zhat_0_from_n = torch.where((t_n == 0).reshape(-1, 1, 1, 1),
                z_0, zhat_0_from_n)
        zhat_0_from_nplus1 = self.student_unet(z_nplus1_scaled, t_nplus1,
            guidance=guidance_scale, encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_mask).sample
        return get_loss(zhat_0_from_nplus1, zhat_0_from_n, gt_wav, prompt,
            t_nplus1, t_ind_nplus1)
