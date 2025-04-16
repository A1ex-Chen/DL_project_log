def prepare_attention_mask(self, attention_mask, target_length, batch_size=
    None, out_dim=3):
    if batch_size is None:
        deprecate('batch_size=None', '0.0.15',
            'Not passing the `batch_size` parameter to `prepare_attention_mask` can lead to incorrect attention mask preparation and is deprecated behavior. Please make sure to pass `batch_size` to `prepare_attention_mask` when preparing the attention_mask.'
            )
        batch_size = 1
    head_size = self.heads
    if attention_mask is None:
        return attention_mask
    current_length: int = attention_mask.shape[-1]
    if current_length != target_length:
        if attention_mask.device.type == 'mps':
            padding_shape = attention_mask.shape[0], attention_mask.shape[1
                ], target_length
            padding = torch.zeros(padding_shape, dtype=attention_mask.dtype,
                device=attention_mask.device)
            attention_mask = torch.cat([attention_mask, padding], dim=2)
        else:
            attention_mask = F.pad(attention_mask, (0, target_length),
                value=0.0)
    if out_dim == 3:
        if attention_mask.shape[0] < batch_size * head_size:
            attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
    elif out_dim == 4:
        attention_mask = attention_mask.unsqueeze(1)
        attention_mask = attention_mask.repeat_interleave(head_size, dim=1)
    return attention_mask
