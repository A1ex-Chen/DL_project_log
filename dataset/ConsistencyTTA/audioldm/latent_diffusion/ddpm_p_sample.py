@torch.no_grad()
def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
    b, *_, device = *x.shape, x.device
    model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t,
        clip_denoised=clip_denoised)
    noise = noise_like(x.shape, device, repeat_noise)
    nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) -
        1))).contiguous()
    return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
