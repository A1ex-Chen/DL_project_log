def get_sample_input(self, batch_size, image_height, image_width):
    latent_height, latent_width = self.check_dims(batch_size, image_height,
        image_width)
    return torch.randn(batch_size, 4, latent_height, latent_width, dtype=
        torch.float32, device=self.device)
