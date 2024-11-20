@apply_forward_hook
def decode(self, z: torch.Tensor, generator: Optional[torch.Generator]=None,
    return_dict: bool=True, num_inference_steps: int=2) ->Union[
    DecoderOutput, Tuple[torch.Tensor]]:
    """
        Decodes the input latent vector `z` using the consistency decoder VAE model.

        Args:
            z (torch.Tensor): The input latent vector.
            generator (Optional[torch.Generator]): The random number generator. Default is None.
            return_dict (bool): Whether to return the output as a dictionary. Default is True.
            num_inference_steps (int): The number of inference steps. Default is 2.

        Returns:
            Union[DecoderOutput, Tuple[torch.Tensor]]: The decoded output.

        """
    z = (z * self.config.scaling_factor - self.means) / self.stds
    scale_factor = 2 ** (len(self.config.block_out_channels) - 1)
    z = F.interpolate(z, mode='nearest', scale_factor=scale_factor)
    batch_size, _, height, width = z.shape
    self.decoder_scheduler.set_timesteps(num_inference_steps, device=self.
        device)
    x_t = self.decoder_scheduler.init_noise_sigma * randn_tensor((
        batch_size, 3, height, width), generator=generator, dtype=z.dtype,
        device=z.device)
    for t in self.decoder_scheduler.timesteps:
        model_input = torch.concat([self.decoder_scheduler.
            scale_model_input(x_t, t), z], dim=1)
        model_output = self.decoder_unet(model_input, t).sample[:, :3, :, :]
        prev_sample = self.decoder_scheduler.step(model_output, t, x_t,
            generator).prev_sample
        x_t = prev_sample
    x_0 = x_t
    if not return_dict:
        return x_0,
    return DecoderOutput(sample=x_0)
