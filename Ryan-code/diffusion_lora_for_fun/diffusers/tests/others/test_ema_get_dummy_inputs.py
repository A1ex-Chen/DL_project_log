def get_dummy_inputs(self):
    noisy_latents = torch.randn(self.batch_size, self.num_in_channels, self
        .latent_height, self.latent_width, generator=self.generator).to(
        torch_device)
    timesteps = torch.randint(0, 1000, size=(self.batch_size,), generator=
        self.generator).to(torch_device)
    encoder_hidden_states = torch.randn(self.batch_size, self.prompt_length,
        self.text_encoder_hidden_dim, generator=self.generator).to(torch_device
        )
    return noisy_latents, timesteps, encoder_hidden_states
