@torch.no_grad()
def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
    verbose=True, timesteps=None, quantize_denoised=False, mask=None, x0=
    None, shape=None, **kwargs):
    if shape is None:
        shape = (batch_size, self.channels, self.latent_t_size, self.
            latent_f_size)
    if cond is not None:
        if isinstance(cond, dict):
            cond = {key: (cond[key][:batch_size] if not isinstance(cond[key
                ], list) else list(map(lambda x: x[:batch_size], cond[key])
                )) for key in cond}
        else:
            cond = [c[:batch_size] for c in cond] if isinstance(cond, list
                ) else cond[:batch_size]
    return self.p_sample_loop(cond, shape, return_intermediates=
        return_intermediates, x_T=x_T, verbose=verbose, timesteps=timesteps,
        quantize_denoised=quantize_denoised, mask=mask, x0=x0, **kwargs)
