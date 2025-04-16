def head_to_batch_dim(self, tensor: torch.Tensor, out_dim: int=3
    ) ->torch.Tensor:
    """
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size, seq_len, heads, dim // heads]` `heads` is
        the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`torch.Tensor`): The tensor to reshape.
            out_dim (`int`, *optional*, defaults to `3`): The output dimension of the tensor. If `3`, the tensor is
                reshaped to `[batch_size * heads, seq_len, dim // heads]`.

        Returns:
            `torch.Tensor`: The reshaped tensor.
        """
    head_size = self.heads
    if tensor.ndim == 3:
        batch_size, seq_len, dim = tensor.shape
        extra_dim = 1
    else:
        batch_size, extra_dim, seq_len, dim = tensor.shape
    tensor = tensor.reshape(batch_size, seq_len * extra_dim, head_size, dim //
        head_size)
    tensor = tensor.permute(0, 2, 1, 3)
    if out_dim == 3:
        tensor = tensor.reshape(batch_size * head_size, seq_len * extra_dim,
            dim // head_size)
    return tensor
