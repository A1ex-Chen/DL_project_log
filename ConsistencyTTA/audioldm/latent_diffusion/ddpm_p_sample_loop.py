@torch.no_grad()
def p_sample_loop(self, shape, return_intermediates=False):
    device = self.betas.device
    b = shape[0]
    img = torch.randn(shape, device=device)
    intermediates = [img]
    for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t',
        total=self.num_timesteps):
        img = self.p_sample(img, torch.full((b,), i, device=device, dtype=
            torch.long), clip_denoised=self.clip_denoised)
        if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
            intermediates.append(img)
    if return_intermediates:
        return img, intermediates
    return img
