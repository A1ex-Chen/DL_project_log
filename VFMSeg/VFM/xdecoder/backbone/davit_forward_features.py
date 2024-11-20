def forward_features(self, x):
    input_size = x.size(2), x.size(3)
    outs = {}
    for i, (conv, block) in enumerate(zip(self.convs, self.blocks)):
        x, input_size = conv(x, input_size)
        if self.enable_checkpoint:
            x, input_size = checkpoint.checkpoint(block, x, input_size)
        else:
            x, input_size = block(x, input_size)
        if i in self.out_indices:
            out = x.view(-1, *input_size, self.embed_dims[i]).permute(0, 3,
                1, 2).contiguous()
            outs['res{}'.format(i + 2)] = out
    if len(self.out_indices) == 0:
        outs['res5'] = x.view(-1, *input_size, self.embed_dims[-1]).permute(
            0, 3, 1, 2).contiguous()
    return outs
