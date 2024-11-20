def _pad_to_window_size(self, input_ids: torch.Tensor, attention_mask:
    torch.Tensor, token_type_ids: torch.Tensor, position_ids: torch.Tensor,
    inputs_embeds: torch.Tensor, pad_token_id: int):
    """A helper function to pad tokens and mask to work with implementation of Longformer self-attention."""
    attention_window = self.config.attention_window if isinstance(self.
        config.attention_window, int) else max(self.config.attention_window)
    assert attention_window % 2 == 0, f'`attention_window` should be an even value. Given {attention_window}'
    input_shape = (input_ids.shape if input_ids is not None else
        inputs_embeds.shape)
    batch_size, seq_len = input_shape[:2]
    padding_len = (attention_window - seq_len % attention_window
        ) % attention_window
    if padding_len > 0:
        logger.info(
            'Input ids are automatically padded from {} to {} to be a multiple of `config.attention_window`: {}'
            .format(seq_len, seq_len + padding_len, attention_window))
        if input_ids is not None:
            input_ids = F.pad(input_ids, (0, padding_len), value=pad_token_id)
        if position_ids is not None:
            position_ids = F.pad(position_ids, (0, padding_len), value=
                pad_token_id)
        if inputs_embeds is not None:
            input_ids_padding = inputs_embeds.new_full((batch_size,
                padding_len), self.config.pad_token_id, dtype=torch.long)
            inputs_embeds_padding = self.embeddings(input_ids_padding)
            inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_padding
                ], dim=-2)
        attention_mask = F.pad(attention_mask, (0, padding_len), value=False)
        token_type_ids = F.pad(token_type_ids, (0, padding_len), value=0)
    return (padding_len, input_ids, attention_mask, token_type_ids,
        position_ids, inputs_embeds)
