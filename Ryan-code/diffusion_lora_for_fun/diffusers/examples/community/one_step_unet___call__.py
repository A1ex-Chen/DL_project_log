def __call__(self):
    image = torch.randn((1, self.unet.config.in_channels, self.unet.config.
        sample_size, self.unet.config.sample_size))
    timestep = 1
    model_output = self.unet(image, timestep).sample
    scheduler_output = self.scheduler.step(model_output, timestep, image
        ).prev_sample
    result = scheduler_output - scheduler_output + torch.ones_like(
        scheduler_output)
    return result
