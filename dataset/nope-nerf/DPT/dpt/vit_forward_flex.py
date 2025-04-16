def forward_flex(self, x):
    b, c, h, w = x.shape
    pos_embed = self._resize_pos_embed(self.pos_embed, h // self.patch_size
        [1], w // self.patch_size[0])
    B = x.shape[0]
    if hasattr(self.patch_embed, 'backbone'):
        x = self.patch_embed.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]
    x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
    if getattr(self, 'dist_token', None) is not None:
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
    else:
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
    x = x + pos_embed
    x = self.pos_drop(x)
    for blk in self.blocks:
        x = blk(x)
    x = self.norm(x)
    return x
