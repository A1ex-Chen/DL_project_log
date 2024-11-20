def reshape_batch_dim_to_heads(self, tensor):
    batch_size, seq_len, dim = tensor.shape
    head_size = self.num_heads
    tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
    tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size,
        seq_len, dim * head_size)
    return tensor
