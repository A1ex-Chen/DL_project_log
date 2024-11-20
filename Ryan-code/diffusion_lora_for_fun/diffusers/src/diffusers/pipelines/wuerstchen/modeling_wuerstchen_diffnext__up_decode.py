def _up_decode(self, level_outputs, r_embed, effnet, clip=None):
    x = level_outputs[0]
    for i, up_block in enumerate(self.up_blocks):
        effnet_c = None
        for j, block in enumerate(up_block):
            if isinstance(block, ResBlockStageB):
                if effnet_c is None and self.effnet_mappers[len(self.
                    down_blocks) + i] is not None:
                    dtype = effnet.dtype
                    effnet_c = self.effnet_mappers[len(self.down_blocks) + i](
                        nn.functional.interpolate(effnet.float(), size=x.
                        shape[-2:], mode='bicubic', antialias=True,
                        align_corners=True).to(dtype))
                skip = level_outputs[i] if j == 0 and i > 0 else None
                if effnet_c is not None:
                    if skip is not None:
                        skip = torch.cat([skip, effnet_c], dim=1)
                    else:
                        skip = effnet_c
                x = block(x, skip)
            elif isinstance(block, AttnBlock):
                x = block(x, clip)
            elif isinstance(block, TimestepBlock):
                x = block(x, r_embed)
            else:
                x = block(x)
    return x
