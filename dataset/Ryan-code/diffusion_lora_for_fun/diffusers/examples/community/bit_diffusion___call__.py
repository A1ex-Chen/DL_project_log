@torch.no_grad()
def __call__(self, height: Optional[int]=256, width: Optional[int]=256,
    num_inference_steps: Optional[int]=50, generator: Optional[torch.
    Generator]=None, batch_size: Optional[int]=1, output_type: Optional[str
    ]='pil', return_dict: bool=True, **kwargs) ->Union[Tuple,
    ImagePipelineOutput]:
    latents = torch.randn((batch_size, self.unet.config.in_channels, height,
        width), generator=generator)
    latents = decimal_to_bits(latents) * self.bit_scale
    latents = latents.to(self.device)
    self.scheduler.set_timesteps(num_inference_steps)
    for t in self.progress_bar(self.scheduler.timesteps):
        noise_pred = self.unet(latents, t).sample
        latents = self.scheduler.step(noise_pred, t, latents).prev_sample
    image = bits_to_decimal(latents)
    if output_type == 'pil':
        image = self.numpy_to_pil(image)
    if not return_dict:
        return image,
    return ImagePipelineOutput(images=image)
