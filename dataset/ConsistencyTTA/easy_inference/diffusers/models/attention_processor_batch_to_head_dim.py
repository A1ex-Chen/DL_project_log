def batch_to_head_dim(self, tensor):
    head_size = self.heads
    batch_size, seq_len, dim = tensor.shape
    tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
    tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size,
        seq_len, dim * head_size)
    return tensor
