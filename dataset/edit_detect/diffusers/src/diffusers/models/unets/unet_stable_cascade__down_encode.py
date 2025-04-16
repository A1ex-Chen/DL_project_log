def _down_encode(self, x, r_embed, clip):
    level_outputs = []
    block_group = zip(self.down_blocks, self.down_downscalers, self.
        down_repeat_mappers)
    if self.training and self.gradient_checkpointing:

        def create_custom_forward(module):

            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        for down_block, downscaler, repmap in block_group:
            x = downscaler(x)
            for i in range(len(repmap) + 1):
                for block in down_block:
                    if isinstance(block, SDCascadeResBlock):
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block), x, use_reentrant=
                            False)
                    elif isinstance(block, SDCascadeAttnBlock):
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block), x, clip,
                            use_reentrant=False)
                    elif isinstance(block, SDCascadeTimestepBlock):
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block), x, r_embed,
                            use_reentrant=False)
                    else:
                        x = x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block), use_reentrant=False)
                if i < len(repmap):
                    x = repmap[i](x)
            level_outputs.insert(0, x)
    else:
        for down_block, downscaler, repmap in block_group:
            x = downscaler(x)
            for i in range(len(repmap) + 1):
                for block in down_block:
                    if isinstance(block, SDCascadeResBlock):
                        x = block(x)
                    elif isinstance(block, SDCascadeAttnBlock):
                        x = block(x, clip)
                    elif isinstance(block, SDCascadeTimestepBlock):
                        x = block(x, r_embed)
                    else:
                        x = block(x)
                if i < len(repmap):
                    x = repmap[i](x)
            level_outputs.insert(0, x)
    return level_outputs
