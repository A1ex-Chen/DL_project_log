def forward(self, input, incremental_state: Optional[Any]=None, timestep:
    Optional[Tensor]=None):
    """Input is expected to be of size [bsz x seqlen]."""
    bsz, seq_len = input.shape[:2]
    max_pos = self.padding_idx + 1 + seq_len
    if max_pos > self.weight.size(0):
        self.make_weight(max_pos, self.embedding_dim, self.padding_idx)
    positions = self.make_positions(input, self.padding_idx)
    return super().forward(positions)
