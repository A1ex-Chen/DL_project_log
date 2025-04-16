@torch.enable_grad()
def cond_fn(self, latents, timestep, index, text_embeddings,
    noise_pred_original, text_embeddings_clip, clip_guidance_scale,
    num_cutouts, use_cutouts=True):
    latents = latents.detach().requires_grad_()
    latent_model_input = self.scheduler.scale_model_input(latents, timestep)
    noise_pred = self.unet(latent_model_input, timestep,
        encoder_hidden_states=text_embeddings).sample
    if isinstance(self.scheduler, (PNDMScheduler, DDIMScheduler,
        DPMSolverMultistepScheduler)):
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (latents - beta_prod_t ** 0.5 * noise_pred
            ) / alpha_prod_t ** 0.5
        fac = torch.sqrt(beta_prod_t)
        sample = pred_original_sample * fac + latents * (1 - fac)
    elif isinstance(self.scheduler, LMSDiscreteScheduler):
        sigma = self.scheduler.sigmas[index]
        sample = latents - sigma * noise_pred
    else:
        raise ValueError(f'scheduler type {type(self.scheduler)} not supported'
            )
    sample = 1 / self.vae.config.scaling_factor * sample
    image = self.vae.decode(sample).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    if use_cutouts:
        image = self.make_cutouts(image, num_cutouts)
    else:
        image = transforms.Resize(self.cut_out_size)(image)
    image = self.normalize(image).to(latents.dtype)
    image_embeddings_clip = self.clip_model.get_image_features(image)
    image_embeddings_clip = image_embeddings_clip / image_embeddings_clip.norm(
        p=2, dim=-1, keepdim=True)
    if use_cutouts:
        dists = spherical_dist_loss(image_embeddings_clip, text_embeddings_clip
            )
        dists = dists.view([num_cutouts, sample.shape[0], -1])
        loss = dists.sum(2).mean(0).sum() * clip_guidance_scale
    else:
        loss = spherical_dist_loss(image_embeddings_clip, text_embeddings_clip
            ).mean() * clip_guidance_scale
    grads = -torch.autograd.grad(loss, latents)[0]
    if isinstance(self.scheduler, LMSDiscreteScheduler):
        latents = latents.detach() + grads * sigma ** 2
        noise_pred = noise_pred_original
    else:
        noise_pred = noise_pred_original - torch.sqrt(beta_prod_t) * grads
    return noise_pred, latents
