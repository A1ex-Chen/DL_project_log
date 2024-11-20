def set_cond_audio(latent_diffusion):
    latent_diffusion.cond_stage_key = 'waveform'
    latent_diffusion.cond_stage_model.embed_mode = 'audio'
    return latent_diffusion
