def set_cond_text(latent_diffusion):
    latent_diffusion.cond_stage_key = 'text'
    latent_diffusion.cond_stage_model.embed_mode = 'text'
    return latent_diffusion
