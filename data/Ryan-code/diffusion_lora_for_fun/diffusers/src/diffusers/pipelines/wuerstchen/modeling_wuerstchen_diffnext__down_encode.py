def _down_encode(self, x, r_embed, effnet, clip=None):
    level_outputs = []
    for i, down_block in enumerate(self.down_blocks):
        effnet_c = None
        for block in down_block:
            if isinstance(block, ResBlockStageB):
                if effnet_c is None and self.effnet_mappers[i] is not None:
                    dtype = effnet.dtype
                    effnet_c = self.effnet_mappers[i](nn.functional.
                        interpolate(effnet.float(), size=x.shape[-2:], mode
                        ='bicubic', antialias=True, align_corners=True).to(
                        dtype))
                skip = effnet_c if self.effnet_mappers[i] is not None else None
                x = block(x, skip)
            elif isinstance(block, AttnBlock):
                x = block(x, clip)
            elif isinstance(block, TimestepBlock):
                x = block(x, r_embed)
            else:
                x = block(x)
        level_outputs.insert(0, x)
    return level_outputs
