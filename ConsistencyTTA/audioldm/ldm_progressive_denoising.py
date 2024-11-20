@torch.no_grad()
def progressive_denoising(self, cond, shape, verbose=True, callback=None,
    quantize_denoised=False, img_callback=None, mask=None, x0=None,
    temperature=1.0, noise_dropout=0.0, score_corrector=None,
    corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
    log_every_t=None):
    if not log_every_t:
        log_every_t = self.log_every_t
    timesteps = self.num_timesteps
    if batch_size is not None:
        b = batch_size if batch_size is not None else shape[0]
        shape = [batch_size] + list(shape)
    else:
        b = batch_size = shape[0]
    if x_T is None:
        img = torch.randn(shape, device=self.device)
    else:
        img = x_T
    intermediates = []
    if cond is not None:
        if isinstance(cond, dict):
            cond = {key: (cond[key][:batch_size] if not isinstance(cond[key
                ], list) else list(map(lambda x: x[:batch_size], cond[key])
                )) for key in cond}
        else:
            cond = [c[:batch_size] for c in cond] if isinstance(cond, list
                ) else cond[:batch_size]
    if start_T is not None:
        timesteps = min(timesteps, start_T)
    iterator = tqdm(reversed(range(0, timesteps)), desc=
        'Progressive Generation', total=timesteps) if verbose else reversed(
        range(0, timesteps))
    if type(temperature) == float:
        temperature = [temperature] * timesteps
    for i in iterator:
        ts = torch.full((b,), i, device=self.device, dtype=torch.long)
        if self.shorten_cond_schedule:
            assert self.model.conditioning_key != 'hybrid'
            tc = self.cond_ids[ts].to(cond.device)
            cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like
                (cond))
        img, x0_partial = self.p_sample(img, cond, ts, clip_denoised=self.
            clip_denoised, quantize_denoised=quantize_denoised, return_x0=
            True, temperature=temperature[i], noise_dropout=noise_dropout,
            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if mask is not None:
            assert x0 is not None
            img_orig = self.q_sample(x0, ts)
            img = img_orig * mask + (1.0 - mask) * img
        if i % log_every_t == 0 or i == timesteps - 1:
            intermediates.append(x0_partial)
        if callback:
            callback(i)
        if img_callback:
            img_callback(img, i)
    return img, intermediates
