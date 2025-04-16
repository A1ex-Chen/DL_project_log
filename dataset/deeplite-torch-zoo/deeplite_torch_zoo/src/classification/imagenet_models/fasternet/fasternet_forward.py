def forward(self, x):
    x = self.patch_embed(x)
    x = self.stages(x)
    x = self.avgpool_pre_head(x)
    x = torch.flatten(x, 1)
    x = self.head(x)
    return x
