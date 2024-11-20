def forward(self, hidden_states: torch.Tensor, encoder_hidden_states:
    Optional[torch.Tensor]=None, image_only_indicator: Optional[torch.
    Tensor]=None, return_dict: bool=True):
    """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch size, channel, height, width)`):
                Input hidden_states.
            num_frames (`int`):
                The number of frames to be processed per batch. This is used to reshape the hidden states.
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            image_only_indicator (`torch.LongTensor` of shape `(batch size, num_frames)`, *optional*):
                A tensor indicating whether the input contains only images. 1 indicates that the input contains only
                images, 0 indicates that the input contains video frames.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_temporal.TransformerTemporalModelOutput`] instead of a
                plain tuple.

        Returns:
            [`~models.transformer_temporal.TransformerTemporalModelOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.transformer_temporal.TransformerTemporalModelOutput`] is
                returned, otherwise a `tuple` where the first element is the sample tensor.
        """
    batch_frames, _, height, width = hidden_states.shape
    num_frames = image_only_indicator.shape[-1]
    batch_size = batch_frames // num_frames
    time_context = encoder_hidden_states
    time_context_first_timestep = time_context[None, :].reshape(batch_size,
        num_frames, -1, time_context.shape[-1])[:, 0]
    time_context = time_context_first_timestep[:, None].broadcast_to(batch_size
        , height * width, time_context.shape[-2], time_context.shape[-1])
    time_context = time_context.reshape(batch_size * height * width, -1,
        time_context.shape[-1])
    residual = hidden_states
    hidden_states = self.norm(hidden_states)
    inner_dim = hidden_states.shape[1]
    hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_frames,
        height * width, inner_dim)
    hidden_states = self.proj_in(hidden_states)
    num_frames_emb = torch.arange(num_frames, device=hidden_states.device)
    num_frames_emb = num_frames_emb.repeat(batch_size, 1)
    num_frames_emb = num_frames_emb.reshape(-1)
    t_emb = self.time_proj(num_frames_emb)
    t_emb = t_emb.to(dtype=hidden_states.dtype)
    emb = self.time_pos_embed(t_emb)
    emb = emb[:, None, :]
    for block, temporal_block in zip(self.transformer_blocks, self.
        temporal_transformer_blocks):
        if self.training and self.gradient_checkpointing:
            hidden_states = torch.utils.checkpoint.checkpoint(block,
                hidden_states, None, encoder_hidden_states, None,
                use_reentrant=False)
        else:
            hidden_states = block(hidden_states, encoder_hidden_states=
                encoder_hidden_states)
        hidden_states_mix = hidden_states
        hidden_states_mix = hidden_states_mix + emb
        hidden_states_mix = temporal_block(hidden_states_mix, num_frames=
            num_frames, encoder_hidden_states=time_context)
        hidden_states = self.time_mixer(x_spatial=hidden_states, x_temporal
            =hidden_states_mix, image_only_indicator=image_only_indicator)
    hidden_states = self.proj_out(hidden_states)
    hidden_states = hidden_states.reshape(batch_frames, height, width,
        inner_dim).permute(0, 3, 1, 2).contiguous()
    output = hidden_states + residual
    if not return_dict:
        return output,
    return TransformerTemporalModelOutput(sample=output)
