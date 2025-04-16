@torch.no_grad()
def _get_eps(self, latent, timestep, guidance_scale, text_embeddings,
    lora_scale=None):
    latent_model_input = torch.cat([latent] * 2
        ) if guidance_scale > 1.0 else latent
    text_embeddings = (text_embeddings if guidance_scale > 1.0 else
        text_embeddings.chunk(2)[1])
    cross_attention_kwargs = None if lora_scale is None else {'scale':
        lora_scale}
    with torch.no_grad():
        noise_pred = self.unet(latent_model_input, timestep,
            encoder_hidden_states=text_embeddings, cross_attention_kwargs=
            cross_attention_kwargs).sample
    if guidance_scale > 1.0:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    elif guidance_scale == 1.0:
        noise_pred_text = noise_pred
        noise_pred_uncond = 0.0
    else:
        raise NotImplementedError(guidance_scale)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text -
        noise_pred_uncond)
    return noise_pred
