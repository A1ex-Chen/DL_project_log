def post_process_latents(self, prior_latents):
    prior_latents = prior_latents * self.clip_std + self.clip_mean
    return prior_latents
