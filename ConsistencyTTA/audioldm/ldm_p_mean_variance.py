def p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids
    =False, quantize_denoised=False, return_x0=False, score_corrector=None,
    corrector_kwargs=None):
    t_in = t
    model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)
    if score_corrector is not None:
        assert self.parameterization == 'eps'
        model_out = score_corrector.modify_score(self, model_out, x, t, c,
            **corrector_kwargs)
    if return_codebook_ids:
        model_out, logits = model_out
    if self.parameterization == 'eps':
        x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
    elif self.parameterization == 'x0':
        x_recon = model_out
    else:
        raise NotImplementedError()
    if clip_denoised:
        x_recon.clamp_(-1.0, 1.0)
    if quantize_denoised:
        x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
    model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
        x_start=x_recon, x_t=x, t=t)
    if return_codebook_ids:
        return model_mean, posterior_variance, posterior_log_variance, logits
    elif return_x0:
        return model_mean, posterior_variance, posterior_log_variance, x_recon
    else:
        return model_mean, posterior_variance, posterior_log_variance
