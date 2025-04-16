def _shape(self, tensor, seq_len, bsz):
    return tensor.contiguous().view(seq_len, bsz * self.num_heads, self.
        head_dim).transpose(0, 1)
