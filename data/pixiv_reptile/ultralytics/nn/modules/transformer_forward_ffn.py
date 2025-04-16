def forward_ffn(self, tgt):
    """Perform forward pass through the Feed-Forward Network part of the layer."""
    tgt2 = self.linear2(self.dropout3(self.act(self.linear1(tgt))))
    tgt = tgt + self.dropout4(tgt2)
    return self.norm3(tgt)
