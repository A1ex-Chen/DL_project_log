def _up_decode(self, level_outputs, r_embed, clip):
    x = level_outputs[0]
    block_group = zip(self.up_blocks, self.up_upscalers, self.up_repeat_mappers
        )
    if self.training and self.gradient_checkpointing:

        def create_custom_forward(module):

            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        for i, (up_block, upscaler, repmap) in enumerate(block_group):
            for j in range(len(repmap) + 1):
                for k, block in enumerate(up_block):
                    if isinstance(block, SDCascadeResBlock):
                        skip = level_outputs[i] if k == 0 and i > 0 else None
                        if skip is not None and (x.size(-1) != skip.size(-1
                            ) or x.size(-2) != skip.size(-2)):
                            orig_type = x.dtype
                            x = torch.nn.functional.interpolate(x.float(),
                                skip.shape[-2:], mode='bilinear',
                                align_corners=True)
                            x = x.to(orig_type)
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block), x, skip,
                            use_reentrant=False)
                    elif isinstance(block, SDCascadeAttnBlock):
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block), x, clip,
                            use_reentrant=False)
                    elif isinstance(block, SDCascadeTimestepBlock):
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block), x, r_embed,
                            use_reentrant=False)
                    else:
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block), x, use_reentrant=
                            False)
                if j < len(repmap):
                    x = repmap[j](x)
            x = upscaler(x)
    else:
        for i, (up_block, upscaler, repmap) in enumerate(block_group):
            for j in range(len(repmap) + 1):
                for k, block in enumerate(up_block):
                    if isinstance(block, SDCascadeResBlock):
                        skip = level_outputs[i] if k == 0 and i > 0 else None
                        if skip is not None and (x.size(-1) != skip.size(-1
                            ) or x.size(-2) != skip.size(-2)):
                            orig_type = x.dtype
                            x = torch.nn.functional.interpolate(x.float(),
                                skip.shape[-2:], mode='bilinear',
                                align_corners=True)
                            x = x.to(orig_type)
                        x = block(x, skip)
                    elif isinstance(block, SDCascadeAttnBlock):
                        x = block(x, clip)
                    elif isinstance(block, SDCascadeTimestepBlock):
                        x = block(x, r_embed)
                    else:
                        x = block(x)
                if j < len(repmap):
                    x = repmap[j](x)
            x = upscaler(x)
    return x
