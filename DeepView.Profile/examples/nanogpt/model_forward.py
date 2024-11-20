def forward(self, idx):
    device = idx.device
    b, t = idx.size()
    assert t <= self.config.block_size, f'Cannot forward sequence of length {t}, block size is only {self.config.block_size}'
    pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
    tok_emb = self.transformer.wte(idx)
    pos_emb = self.transformer.wpe(pos)
    x = self.transformer.drop(tok_emb + pos_emb)
    for block in self.transformer.h:
        x = block(x)
    x = self.transformer.ln_f(x)
    logits = self.lm_head(x)
    return logits
