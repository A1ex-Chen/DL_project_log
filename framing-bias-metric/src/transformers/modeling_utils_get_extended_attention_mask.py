def get_extended_attention_mask(self, attention_mask: Tensor, input_shape:
    Tuple[int], device: device) ->Tensor:
    """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        if self.config.is_decoder:
            batch_size, seq_length = input_shape
            seq_ids = torch.arange(seq_length, device=device)
            causal_mask = seq_ids[None, None, :].repeat(batch_size,
                seq_length, 1) <= seq_ids[None, :, None]
            causal_mask = causal_mask.to(attention_mask.dtype)
            if causal_mask.shape[1] < attention_mask.shape[1]:
                prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                causal_mask = torch.cat([torch.ones((batch_size, seq_length,
                    prefix_seq_len), device=device, dtype=causal_mask.dtype
                    ), causal_mask], axis=-1)
            extended_attention_mask = causal_mask[:, None, :, :
                ] * attention_mask[:, None, None, :]
        else:
            extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            'Wrong shape for input_ids (shape {}) or attention_mask (shape {})'
            .format(input_shape, attention_mask.shape))
    extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask
