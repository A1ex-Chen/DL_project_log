def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
    """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
    if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1
            ).unsqueeze(-1)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    assert head_mask.dim(
        ) == 5, f'head_mask.dim != 5, instead {head_mask.dim()}'
    head_mask = head_mask.to(dtype=self.dtype)
    return head_mask
