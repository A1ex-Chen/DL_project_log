def _freeze_stages(self):
    if self.frozen_stages >= 0:
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False
    if self.frozen_stages >= 2:
        self.pos_drop.eval()
        for i in range(0, self.frozen_stages - 1):
            m = self.layers[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
