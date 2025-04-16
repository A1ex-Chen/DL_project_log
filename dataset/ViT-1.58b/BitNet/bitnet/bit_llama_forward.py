@torch.inference_mode()
def forward(self, tokens: torch.Tensor, start_pos: int):
    """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.
            start_pos (int): Starting position for attention caching.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
    _bsz, seqlen = tokens.shape
    h = self.tok_embeddings(tokens)
    self.freqs_cis = self.freqs_cis.to(h.device)
    freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
    mask = None
    if seqlen > 1:
        mask = torch.full((seqlen, seqlen), float('-inf'), device=tokens.device
            )
        mask = torch.triu(mask, diagonal=1)
        mask = torch.hstack([torch.zeros((seqlen, start_pos), device=tokens
            .device), mask]).type_as(h)
    for layer in self.layers:
        h = layer(h, start_pos, freqs_cis, mask)
    h = self.norm(h)
    output = self.output(h).float()
    return output
