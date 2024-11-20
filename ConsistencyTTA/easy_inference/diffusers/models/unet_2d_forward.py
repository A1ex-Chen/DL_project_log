def forward(self, sample: torch.FloatTensor, timestep: Union[torch.Tensor,
    float, int], class_labels: Optional[torch.Tensor]=None, return_dict:
    bool=True) ->Union[UNet2DOutput, Tuple]:
    """
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int): (batch) timesteps
            class_labels (`torch.FloatTensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d.UNet2DOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d.UNet2DOutput`] or `tuple`: [`~models.unet_2d.UNet2DOutput`] if `return_dict` is True,
            otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.
        """
    if self.config.center_input_sample:
        sample = 2 * sample - 1.0
    timesteps = timestep
    if not torch.is_tensor(timesteps):
        timesteps = torch.tensor([timesteps], dtype=torch.long, device=
            sample.device)
    elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
        timesteps = timesteps[None].to(sample.device)
    timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.
        dtype, device=timesteps.device)
    t_emb = self.time_proj(timesteps)
    t_emb = t_emb.to(dtype=self.dtype)
    emb = self.time_embedding(t_emb)
    if self.class_embedding is not None:
        if class_labels is None:
            raise ValueError(
                'class_labels should be provided when doing class conditioning'
                )
        if self.config.class_embed_type == 'timestep':
            class_labels = self.time_proj(class_labels)
        class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
        emb = emb + class_emb
    skip_sample = sample
    sample = self.conv_in(sample)
    down_block_res_samples = sample,
    for downsample_block in self.down_blocks:
        if hasattr(downsample_block, 'skip_conv'):
            sample, res_samples, skip_sample = downsample_block(hidden_states
                =sample, temb=emb, skip_sample=skip_sample)
        else:
            sample, res_samples = downsample_block(hidden_states=sample,
                temb=emb)
        down_block_res_samples += res_samples
    sample = self.mid_block(sample, emb)
    skip_sample = None
    for upsample_block in self.up_blocks:
        res_samples = down_block_res_samples[-len(upsample_block.resnets):]
        down_block_res_samples = down_block_res_samples[:-len(
            upsample_block.resnets)]
        if hasattr(upsample_block, 'skip_conv'):
            sample, skip_sample = upsample_block(sample, res_samples, emb,
                skip_sample)
        else:
            sample = upsample_block(sample, res_samples, emb)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    if skip_sample is not None:
        sample += skip_sample
    if self.config.time_embedding_type == 'fourier':
        timesteps = timesteps.reshape((sample.shape[0], *([1] * len(sample.
            shape[1:]))))
        sample = sample / timesteps
    if not return_dict:
        return sample,
    return UNet2DOutput(sample=sample)
