def forward(self, x, register_blk=-1):
    B = x.shape[0]
    x = self.patch_embed(x)
    cls_tokens = self.cls_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    x = x + self.pos_embed[:, :x.size(1), :]
    x = self.pos_drop(x)
    for i, blk in enumerate(self.blocks):
        x = blk(x, register_blk == i)
    x = self.norm(x)
    return x
