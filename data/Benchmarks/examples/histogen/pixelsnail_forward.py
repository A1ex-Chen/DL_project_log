def forward(self, input, condition=None, cache=None):
    if cache is None:
        cache = {}
    batch, height, width = input.shape
    input = F.one_hot(input, self.n_class).permute(0, 3, 1, 2).type_as(self
        .background)
    horizontal = shift_down(self.horizontal(input))
    vertical = shift_right(self.vertical(input))
    out = horizontal + vertical
    background = self.background[:, :, :height, :].expand(batch, 2, height,
        width)
    if condition is not None:
        if 'condition' in cache:
            condition = cache['condition']
            condition = condition[:, :, :height, :]
        else:
            condition = F.one_hot(condition, self.n_class).permute(0, 3, 1, 2
                ).type_as(self.background)
            condition = self.cond_resnet(condition)
            condition = F.interpolate(condition, scale_factor=2)
            cache['condition'] = condition.detach().clone()
            condition = condition[:, :, :height, :]
    for block in self.blocks:
        out = block(out, background, condition=condition)
    out = self.out(out)
    return out, cache
