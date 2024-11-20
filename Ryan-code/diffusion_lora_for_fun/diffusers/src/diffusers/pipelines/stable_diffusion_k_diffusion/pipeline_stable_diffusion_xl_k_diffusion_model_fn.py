def model_fn(x, t):
    latent_model_input = torch.cat([x] * 2)
    t = torch.cat([t] * 2)
    noise_pred = self.k_diffusion_model(latent_model_input, t, cond=
        prompt_embeds, timestep_cond=timestep_cond, added_cond_kwargs=
        added_cond_kwargs)
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text -
        noise_pred_uncond)
    return noise_pred
