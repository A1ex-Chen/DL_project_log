@torch.no_grad()
def ddim_sampling(self, cond, shape, x_T=None, ddim_use_original_steps=
    False, callback=None, timesteps=None, quantize_denoised=False, mask=
    None, x0=None, img_callback=None, log_every_t=100, temperature=1.0,
    noise_dropout=0.0, score_corrector=None, corrector_kwargs=None,
    unconditional_guidance_scale=1.0, unconditional_conditioning=None):
    device = self.model.betas.device
    b = shape[0]
    if x_T is None:
        img = torch.randn(shape, device=device)
    else:
        img = x_T
    if timesteps is None:
        timesteps = (self.ddpm_num_timesteps if ddim_use_original_steps else
            self.ddim_timesteps)
    elif timesteps is not None and not ddim_use_original_steps:
        subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) *
            self.ddim_timesteps.shape[0]) - 1
        timesteps = self.ddim_timesteps[:subset_end]
    intermediates = {'x_inter': [img], 'pred_x0': [img]}
    time_range = reversed(range(0, timesteps)
        ) if ddim_use_original_steps else np.flip(timesteps)
    total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
    print(f'Running DDIM Sampling with {total_steps} timesteps')
    iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
    for i, step in enumerate(iterator):
        index = total_steps - i - 1
        ts = torch.full((b,), step, device=device, dtype=torch.long)
        if mask is not None:
            assert x0 is not None
            img_orig = self.model.q_sample(x0, ts)
            img = img_orig * mask + (1.0 - mask) * img
        outs = self.p_sample_ddim(img, cond, ts, index=index,
            use_original_steps=ddim_use_original_steps, quantize_denoised=
            quantize_denoised, temperature=temperature, noise_dropout=
            noise_dropout, score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs, unconditional_guidance_scale
            =unconditional_guidance_scale, unconditional_conditioning=
            unconditional_conditioning)
        img, pred_x0 = outs
        if callback:
            callback(i)
        if img_callback:
            img_callback(pred_x0, i)
        if index % log_every_t == 0 or index == total_steps - 1:
            intermediates['x_inter'].append(img)
            intermediates['pred_x0'].append(pred_x0)
    return img, intermediates
