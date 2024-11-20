def preprocess_feedback_images(self, images, vae, dim, device, dtype, generator
    ) ->torch.tensor:
    images_t = [self.image_to_tensor(img, dim, dtype) for img in images]
    images_t = torch.stack(images_t).to(device)
    latents = vae.config.scaling_factor * vae.encode(images_t
        ).latent_dist.sample(generator)
    return torch.cat([latents], dim=0)
