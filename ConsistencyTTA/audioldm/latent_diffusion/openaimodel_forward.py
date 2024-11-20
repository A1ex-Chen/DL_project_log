def forward(self, x, timesteps):
    """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
    emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
    results = []
    h = x.type(self.dtype)
    for module in self.input_blocks:
        h = module(h, emb)
        if self.pool.startswith('spatial'):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
    h = self.middle_block(h, emb)
    if self.pool.startswith('spatial'):
        results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = th.cat(results, axis=-1)
        return self.out(h)
    else:
        h = h.type(x.dtype)
        return self.out(h)
