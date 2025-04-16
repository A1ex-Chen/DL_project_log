def reshape_heads_to_batch_dim(self, tensor):
    batch_size, seq_len, dim = tensor.shape
    head_size = self.heads
    tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
    tensor = jnp.transpose(tensor, (0, 2, 1, 3))
    tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size)
    return tensor
