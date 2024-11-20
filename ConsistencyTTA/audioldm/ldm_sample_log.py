@torch.no_grad()
def sample_log(self, cond, batch_size, ddim, ddim_steps,
    unconditional_guidance_scale=1.0, unconditional_conditioning=None,
    use_plms=False, mask=None, **kwargs):
    if mask is not None:
        shape = self.channels, mask.size()[-2], mask.size()[-1]
    else:
        shape = self.channels, self.latent_t_size, self.latent_f_size
    intermediate = None
    if ddim and not use_plms:
        ddim_sampler = DDIMSampler(self)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size,
            shape, cond, verbose=False, unconditional_guidance_scale=
            unconditional_guidance_scale, unconditional_conditioning=
            unconditional_conditioning, mask=mask, **kwargs)
    else:
        samples, intermediates = self.sample(cond=cond, batch_size=
            batch_size, return_intermediates=True,
            unconditional_guidance_scale=unconditional_guidance_scale, mask
            =mask, unconditional_conditioning=unconditional_conditioning,
            **kwargs)
    return samples, intermediate
