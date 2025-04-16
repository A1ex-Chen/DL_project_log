def forward(self, hidden_states, encoder_hidden_states=None, timestep=None,
    class_labels=None, num_frames=1, cross_attention_kwargs=None,
    return_dict: bool=True):
    """
        Args:
            hidden_states ( When discrete, `torch.LongTensor` of shape `(batch size, num latent pixels)`.
                When continous, `torch.FloatTensor` of shape `(batch size, channel, height, width)`): Input
                hidden_states
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.long`, *optional*):
                Optional timestep to be applied as an embedding in AdaLayerNorm's. Used to indicate denoising step.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Optional class labels to be applied as an embedding in AdaLayerZeroNorm. Used to indicate class labels
                conditioning.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.transformer_2d.TransformerTemporalModelOutput`] or `tuple`:
            [`~models.transformer_2d.TransformerTemporalModelOutput`] if `return_dict` is True, otherwise a `tuple`.
            When returning a tuple, the first element is the sample tensor.
        """
    batch_frames, channel, height, width = hidden_states.shape
    batch_size = batch_frames // num_frames
    residual = hidden_states
    hidden_states = hidden_states[None, :].reshape(batch_size, num_frames,
        channel, height, width)
    hidden_states = hidden_states.permute(0, 2, 1, 3, 4)
    hidden_states = self.norm(hidden_states)
    hidden_states = hidden_states.permute(0, 3, 4, 2, 1).reshape(batch_size *
        height * width, num_frames, channel)
    hidden_states = self.proj_in(hidden_states)
    for block in self.transformer_blocks:
        hidden_states = block(hidden_states, encoder_hidden_states=
            encoder_hidden_states, timestep=timestep,
            cross_attention_kwargs=cross_attention_kwargs, class_labels=
            class_labels)
    hidden_states = self.proj_out(hidden_states)
    hidden_states = hidden_states[None, None, :].reshape(batch_size, height,
        width, channel, num_frames).permute(0, 3, 4, 1, 2).contiguous()
    hidden_states = hidden_states.reshape(batch_frames, channel, height, width)
    output = hidden_states + residual
    if not return_dict:
        return output,
    return TransformerTemporalModelOutput(sample=output)
