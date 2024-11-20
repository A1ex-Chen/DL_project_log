@torch.no_grad()
def sample(self, S, batch_size, shape, conditioning=None, callback=None,
    normals_sequence=None, img_callback=None, quantize_x0=False, eta=0.0,
    mask=None, x0=None, temperature=1.0, noise_dropout=0.0, score_corrector
    =None, corrector_kwargs=None, verbose=True, x_T=None, log_every_t=100,
    unconditional_guidance_scale=1.0, unconditional_conditioning=None, **kwargs
    ):
    if conditioning is not None:
        if isinstance(conditioning, dict):
            cbs = conditioning[list(conditioning.keys())[0]].shape[0]
            if cbs != batch_size:
                print(
                    f'Warning: Got {cbs} conditionings but batch-size is {batch_size}'
                    )
        elif conditioning.shape[0] != batch_size:
            print(
                f'Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}'
                )
    self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
    C, H, W = shape
    size = batch_size, C, H, W
    print(f'Data shape for DDIM sampling is {size}, eta {eta}')
    samples, intermediates = self.ddim_sampling(conditioning, size,
        callback=callback, img_callback=img_callback, quantize_denoised=
        quantize_x0, mask=mask, x0=x0, ddim_use_original_steps=False,
        noise_dropout=noise_dropout, temperature=temperature,
        score_corrector=score_corrector, corrector_kwargs=corrector_kwargs,
        x_T=x_T, log_every_t=log_every_t, unconditional_guidance_scale=
        unconditional_guidance_scale, unconditional_conditioning=
        unconditional_conditioning)
    return samples, intermediates
