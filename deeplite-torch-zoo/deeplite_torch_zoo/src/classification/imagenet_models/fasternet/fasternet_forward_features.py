def forward_features(self, x: Tensor) ->Tensor:
    x = self.patch_embed(x)
    outs = []
    for idx, stage in enumerate(self.stages):
        x = stage(x)
        if self.fork_feat and idx in self.out_indices:
            norm_layer = getattr(self, f'norm{idx}')
            x_out = norm_layer(x)
            outs.append(x_out)
    return outs
