def _intermediate_layers(self, x: torch.Tensor, n: Union[int, Sequence]=1
    ) ->List[torch.Tensor]:
    outputs, num_blocks = [], len(self.blocks)
    take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n,
        int) else n)
    x = self.patch_embed(x)
    x = self._pos_embed(x)
    x = self.patch_drop(x)
    x = self.norm_pre(x)
    for i, blk in enumerate(self.blocks):
        x = blk(x)
        if i in take_indices:
            outputs.append(x)
    return outputs
