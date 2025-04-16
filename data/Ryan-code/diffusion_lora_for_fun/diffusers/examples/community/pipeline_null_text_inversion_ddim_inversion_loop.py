@torch.no_grad()
def ddim_inversion_loop(self, latent, context):
    self.scheduler.set_timesteps(self.num_inference_steps)
    _, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    with torch.no_grad():
        for i in range(0, self.num_inference_steps):
            t = self.scheduler.timesteps[len(self.scheduler.timesteps) - i - 1]
            noise_pred = self.unet(latent, t, encoder_hidden_states=
                cond_embeddings)['sample']
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
    return all_latent
