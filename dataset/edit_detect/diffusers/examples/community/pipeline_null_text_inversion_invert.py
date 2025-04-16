def invert(self, image_path: str, prompt: str, num_inner_steps=10,
    early_stop_epsilon=1e-06, num_inference_steps=50):
    self.num_inference_steps = num_inference_steps
    context = self.get_context(prompt)
    latent = self.image2latent(image_path)
    ddim_latents = self.ddim_inversion_loop(latent, context)
    if os.path.exists(image_path + '.pt'):
        uncond_embeddings = torch.load(image_path + '.pt')
    else:
        uncond_embeddings = self.null_optimization(ddim_latents, context,
            num_inner_steps, early_stop_epsilon)
        uncond_embeddings = torch.stack(uncond_embeddings, 0)
        torch.save(uncond_embeddings, image_path + '.pt')
    return ddim_latents[-1], uncond_embeddings
