def _pad_tensors_to_max_len(self, tensor, max_length):
    pad_token_id = (self.config.pad_token_id if self.config.pad_token_id is not
        None else self.config.eos_token_id)
    if pad_token_id is None:
        raise ValueError(
            f'Make sure that either `config.pad_token_id` or `config.eos_token_id` is defined if tensor has to be padded to `max_length`={max_length}'
            )
    padded_tensor = pad_token_id * torch.ones((tensor.shape[0], max_length),
        dtype=tensor.dtype, device=tensor.device)
    padded_tensor[:, :tensor.shape[-1]] = tensor
    return padded_tensor
