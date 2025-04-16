def compute_vae_encodings(batch, vae):
    images = batch.pop('pixel_values')
    pixel_values = torch.stack(list(images))
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format
        ).float()
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)
    with torch.no_grad():
        model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = model_input * vae.config.scaling_factor
    return {'model_input': model_input.cpu()}
