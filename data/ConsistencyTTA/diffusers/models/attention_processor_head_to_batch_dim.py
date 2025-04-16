def head_to_batch_dim(self, tensor, out_dim=3):
    head_size = self.heads
    batch_size, seq_len, dim = tensor.shape
    tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
    tensor = tensor.permute(0, 2, 1, 3)
    if out_dim == 3:
        tensor = tensor.reshape(batch_size * head_size, seq_len, dim //
            head_size)
    return tensor
