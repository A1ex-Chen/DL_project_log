@torch.no_grad()
def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
    return_codebook_ids=False, quantize_denoised=False, return_x0=False,
    temperature=1.0, noise_dropout=0.0, score_corrector=None,
    corrector_kwargs=None):
    b, *_, device = *x.shape, x.device
    outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=
        clip_denoised, return_codebook_ids=return_codebook_ids,
        quantize_denoised=quantize_denoised, return_x0=return_x0,
        score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
    if return_codebook_ids:
        raise DeprecationWarning('Support dropped.')
        model_mean, _, model_log_variance, logits = outputs
    elif return_x0:
        model_mean, _, model_log_variance, x0 = outputs
    else:
        model_mean, _, model_log_variance = outputs
    noise = noise_like(x.shape, device, repeat_noise) * temperature
    if noise_dropout > 0.0:
        noise = torch.nn.functional.dropout(noise, p=noise_dropout)
    nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) -
        1))).contiguous()
    if return_codebook_ids:
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp(
            ) * noise, logits.argmax(dim=1)
    if return_x0:
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp(
            ) * noise, x0
    else:
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp(
            ) * noise
