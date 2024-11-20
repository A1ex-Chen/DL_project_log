@torch.set_grad_enabled(True)
def latent_lmd_guidance(self, cond_embeddings, index, boxes, phrase_indices,
    t, latents, loss, *, loss_scale=20, loss_threshold=5.0, max_iter=[3] * 
    5 + [2] * 5 + [1] * 5, guidance_timesteps=15, cross_attention_kwargs=
    None, guidance_attn_keys=DEFAULT_GUIDANCE_ATTN_KEYS, verbose=False,
    clear_cache=False, unet_additional_kwargs={}, guidance_callback=None,
    **kwargs):
    scheduler, unet = self.scheduler, self.unet
    iteration = 0
    if index < guidance_timesteps:
        if isinstance(max_iter, list):
            max_iter = max_iter[index]
        if verbose:
            logger.info(
                f'time index {index}, loss: {loss.item() / loss_scale:.3f} (de-scaled with scale {loss_scale:.1f}), loss threshold: {loss_threshold:.3f}'
                )
        try:
            self.enable_attn_hook(enabled=True)
            while (loss.item() / loss_scale > loss_threshold and iteration <
                max_iter and index < guidance_timesteps):
                self._saved_attn = {}
                latents.requires_grad_(True)
                latent_model_input = latents
                latent_model_input = scheduler.scale_model_input(
                    latent_model_input, t)
                unet(latent_model_input, t, encoder_hidden_states=
                    cond_embeddings, cross_attention_kwargs=
                    cross_attention_kwargs, **unet_additional_kwargs)
                loss = self.compute_ca_loss(saved_attn=self._saved_attn,
                    bboxes=boxes, phrase_indices=phrase_indices,
                    guidance_attn_keys=guidance_attn_keys, verbose=verbose,
                    **kwargs) * loss_scale
                if torch.isnan(loss):
                    raise RuntimeError('**Loss is NaN**')
                if guidance_callback is not None:
                    guidance_callback(self, latents, loss, iteration, index)
                self._saved_attn = None
                grad_cond = torch.autograd.grad(loss.requires_grad_(True),
                    [latents])[0]
                latents.requires_grad_(False)
                alpha_prod_t = scheduler.alphas_cumprod[t]
                scale = (1 - alpha_prod_t) ** 0.5
                latents = latents - scale * grad_cond
                iteration += 1
                if clear_cache:
                    gc.collect()
                    torch.cuda.empty_cache()
                if verbose:
                    logger.info(
                        f'time index {index}, loss: {loss.item() / loss_scale:.3f}, loss threshold: {loss_threshold:.3f}, iteration: {iteration}'
                        )
        finally:
            self.enable_attn_hook(enabled=False)
    return latents, loss
