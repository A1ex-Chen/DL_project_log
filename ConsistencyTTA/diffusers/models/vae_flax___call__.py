def __call__(self, sample, sample_posterior=False, deterministic: bool=True,
    return_dict: bool=True):
    posterior = self.encode(sample, deterministic=deterministic,
        return_dict=return_dict)
    if sample_posterior:
        rng = self.make_rng('gaussian')
        hidden_states = posterior.latent_dist.sample(rng)
    else:
        hidden_states = posterior.latent_dist.mode()
    sample = self.decode(hidden_states, return_dict=return_dict).sample
    if not return_dict:
        return sample,
    return FlaxDecoderOutput(sample=sample)
