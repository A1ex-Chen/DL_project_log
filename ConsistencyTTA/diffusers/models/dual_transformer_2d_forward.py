def forward(self, hidden_states, encoder_hidden_states, timestep=None,
    attention_mask=None, cross_attention_kwargs=None, return_dict: bool=True):
    """
        Args:
            hidden_states ( When discrete, `torch.LongTensor` of shape `(batch size, num latent pixels)`.
                When continuous, `torch.FloatTensor` of shape `(batch size, channel, height, width)`): Input
                hidden_states
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.long`, *optional*):
                Optional timestep to be applied as an embedding in AdaLayerNorm's. Used to indicate denoising step.
            attention_mask (`torch.FloatTensor`, *optional*):
                Optional attention mask to be applied in Attention
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.transformer_2d.Transformer2DModelOutput`] or `tuple`:
            [`~models.transformer_2d.Transformer2DModelOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
    input_states = hidden_states
    encoded_states = []
    tokens_start = 0
    for i in range(2):
        condition_state = encoder_hidden_states[:, tokens_start:
            tokens_start + self.condition_lengths[i]]
        transformer_index = self.transformer_index_for_condition[i]
        encoded_state = self.transformers[transformer_index](input_states,
            encoder_hidden_states=condition_state, timestep=timestep,
            cross_attention_kwargs=cross_attention_kwargs, return_dict=False)[0
            ]
        encoded_states.append(encoded_state - input_states)
        tokens_start += self.condition_lengths[i]
    output_states = encoded_states[0] * self.mix_ratio + encoded_states[1] * (
        1 - self.mix_ratio)
    output_states = output_states + input_states
    if not return_dict:
        return output_states,
    return Transformer2DModelOutput(sample=output_states)
