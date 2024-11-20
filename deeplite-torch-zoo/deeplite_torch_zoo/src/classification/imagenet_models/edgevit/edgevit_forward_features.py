def forward_features(self, x):
    x = self.patch_embed1(x)
    x = self.pos_drop(x)
    for blk in self.blocks1:
        x = blk(x)
    x = self.patch_embed2(x)
    for blk in self.blocks2:
        x = blk(x)
    x = self.patch_embed3(x)
    for blk in self.blocks3:
        x = blk(x)
    x = self.patch_embed4(x)
    for blk in self.blocks4:
        x = blk(x)
    x = self.norm(x)
    x = self.pre_logits(x)
    return x
