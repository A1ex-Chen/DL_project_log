def forward(self, sample: torch.Tensor, timestep: Union[torch.Tensor, float,
    int], return_dict: bool=True) ->Union[UNet1DOutput, Tuple]:
    """
        The [`UNet1DModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch_size, num_channels, sample_size)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_1d.UNet1DOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_1d.UNet1DOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_1d.UNet1DOutput`] is returned, otherwise a `tuple` is
                returned where the first element is the sample tensor.
        """
    timesteps = timestep
    if not torch.is_tensor(timesteps):
        timesteps = torch.tensor([timesteps], dtype=torch.long, device=
            sample.device)
    elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
        timesteps = timesteps[None].to(sample.device)
    timestep_embed = self.time_proj(timesteps)
    if self.config.use_timestep_embedding:
        timestep_embed = self.time_mlp(timestep_embed)
    else:
        timestep_embed = timestep_embed[..., None]
        timestep_embed = timestep_embed.repeat([1, 1, sample.shape[2]]).to(
            sample.dtype)
        timestep_embed = timestep_embed.broadcast_to(sample.shape[:1] +
            timestep_embed.shape[1:])
    down_block_res_samples = ()
    for downsample_block in self.down_blocks:
        sample, res_samples = downsample_block(hidden_states=sample, temb=
            timestep_embed)
        down_block_res_samples += res_samples
    if self.mid_block:
        sample = self.mid_block(sample, timestep_embed)
    for i, upsample_block in enumerate(self.up_blocks):
        res_samples = down_block_res_samples[-1:]
        down_block_res_samples = down_block_res_samples[:-1]
        sample = upsample_block(sample, res_hidden_states_tuple=res_samples,
            temb=timestep_embed)
    if self.out_block:
        sample = self.out_block(sample, timestep_embed)
    if not return_dict:
        return sample,
    return UNet1DOutput(sample=sample)
