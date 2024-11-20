@torch.inference_mode()
def forward(self, tokens: torch.Tensor, start_pos: int):
    _bsz, seqlen = tokens.shape
    h = self.tok_embeddings(tokens)
    self.freqs_cis = self.freqs_cis.to(h.device)
    freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
    mask = None
    if seqlen > 1:
        mask = torch.full((1, 1, seqlen, seqlen), float('-inf'), device=
            tokens.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
    for layer in self.layers:
        h = layer(h, start_pos, freqs_cis, mask)
    h = self.norm(h)
    output = self.output(h[:, -1, :])
    return output.float()
