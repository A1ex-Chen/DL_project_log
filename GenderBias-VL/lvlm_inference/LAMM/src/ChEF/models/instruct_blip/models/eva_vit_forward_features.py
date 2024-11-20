def forward_features(self, x):
    x = self.patch_embed(x)
    batch_size, seq_len, _ = x.size()
    cls_tokens = self.cls_token.expand(batch_size, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    if self.pos_embed is not None:
        x = x + self.pos_embed
    x = self.pos_drop(x)
    rel_pos_bias = self.rel_pos_bias(
        ) if self.rel_pos_bias is not None else None
    for blk in self.blocks:
        if self.use_checkpoint:
            x = checkpoint.checkpoint(blk, x, rel_pos_bias)
        else:
            x = blk(x, rel_pos_bias)
    return x
