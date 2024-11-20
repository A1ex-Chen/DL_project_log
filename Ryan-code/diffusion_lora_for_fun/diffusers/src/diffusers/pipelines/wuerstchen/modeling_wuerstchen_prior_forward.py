def forward(self, x, r, c):
    x_in = x
    x = self.projection(x)
    c_embed = self.cond_mapper(c)
    r_embed = self.gen_r_embedding(r)
    if self.training and self.gradient_checkpointing:

        def create_custom_forward(module):

            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        if is_torch_version('>=', '1.11.0'):
            for block in self.blocks:
                if isinstance(block, AttnBlock):
                    x = torch.utils.checkpoint.checkpoint(create_custom_forward
                        (block), x, c_embed, use_reentrant=False)
                elif isinstance(block, TimestepBlock):
                    x = torch.utils.checkpoint.checkpoint(create_custom_forward
                        (block), x, r_embed, use_reentrant=False)
                else:
                    x = torch.utils.checkpoint.checkpoint(create_custom_forward
                        (block), x, use_reentrant=False)
        else:
            for block in self.blocks:
                if isinstance(block, AttnBlock):
                    x = torch.utils.checkpoint.checkpoint(create_custom_forward
                        (block), x, c_embed)
                elif isinstance(block, TimestepBlock):
                    x = torch.utils.checkpoint.checkpoint(create_custom_forward
                        (block), x, r_embed)
                else:
                    x = torch.utils.checkpoint.checkpoint(create_custom_forward
                        (block), x)
    else:
        for block in self.blocks:
            if isinstance(block, AttnBlock):
                x = block(x, c_embed)
            elif isinstance(block, TimestepBlock):
                x = block(x, r_embed)
            else:
                x = block(x)
    a, b = self.out(x).chunk(2, dim=1)
    return (x_in - a) / ((1 - b).abs() + 1e-05)
