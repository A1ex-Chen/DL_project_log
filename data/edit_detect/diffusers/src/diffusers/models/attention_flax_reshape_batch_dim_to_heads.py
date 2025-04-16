def reshape_batch_dim_to_heads(self, tensor):
    batch_size, seq_len, dim = tensor.shape
    head_size = self.heads
    tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
    tensor = jnp.transpose(tensor, (0, 2, 1, 3))
    tensor = tensor.reshape(batch_size // head_size, seq_len, dim * head_size)
    return tensor
