@torch.no_grad()
def p_sample_loop(self, cond, shape, return_intermediates=False, x_T=None,
    verbose=True, callback=None, timesteps=None, quantize_denoised=False,
    mask=None, x0=None, img_callback=None, start_T=None, log_every_t=None):
    if not log_every_t:
        log_every_t = self.log_every_t
    device = self.betas.device
    b = shape[0]
    if x_T is None:
        img = torch.randn(shape, device=device)
    else:
        img = x_T
    intermediates = [img]
    if timesteps is None:
        timesteps = self.num_timesteps
    if start_T is not None:
        timesteps = min(timesteps, start_T)
    iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total
        =timesteps) if verbose else reversed(range(0, timesteps))
    if mask is not None:
        assert x0 is not None
        assert x0.shape[2:3] == mask.shape[2:3]
    for i in iterator:
        ts = torch.full((b,), i, device=device, dtype=torch.long)
        if self.shorten_cond_schedule:
            assert self.model.conditioning_key != 'hybrid'
            tc = self.cond_ids[ts].to(cond.device)
            cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like
                (cond))
        img = self.p_sample(img, cond, ts, clip_denoised=self.clip_denoised,
            quantize_denoised=quantize_denoised)
        if mask is not None:
            img_orig = self.q_sample(x0, ts)
            img = img_orig * mask + (1.0 - mask) * img
        if i % log_every_t == 0 or i == timesteps - 1:
            intermediates.append(img)
        if callback:
            callback(i)
        if img_callback:
            img_callback(img, i)
    if return_intermediates:
        return img, intermediates
    return img
