def retrieve_latents(encoder_output: torch.Tensor, generator: Optional[
    torch.Generator]=None, sample_mode: str='sample'):
    if hasattr(encoder_output, 'latent_dist') and sample_mode == 'sample':
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, 'latent_dist') and sample_mode == 'argmax':
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, 'latents'):
        return encoder_output.latents
    else:
        raise AttributeError(
            'Could not access latents of provided encoder_output')
