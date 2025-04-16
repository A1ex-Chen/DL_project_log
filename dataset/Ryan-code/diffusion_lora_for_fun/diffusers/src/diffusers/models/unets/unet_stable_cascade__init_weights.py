def _init_weights(self, m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    nn.init.normal_(self.clip_txt_pooled_mapper.weight, std=0.02)
    nn.init.normal_(self.clip_txt_mapper.weight, std=0.02) if hasattr(self,
        'clip_txt_mapper') else None
    nn.init.normal_(self.clip_img_mapper.weight, std=0.02) if hasattr(self,
        'clip_img_mapper') else None
    if hasattr(self, 'effnet_mapper'):
        nn.init.normal_(self.effnet_mapper[0].weight, std=0.02)
        nn.init.normal_(self.effnet_mapper[2].weight, std=0.02)
    if hasattr(self, 'pixels_mapper'):
        nn.init.normal_(self.pixels_mapper[0].weight, std=0.02)
        nn.init.normal_(self.pixels_mapper[2].weight, std=0.02)
    torch.nn.init.xavier_uniform_(self.embedding[1].weight, 0.02)
    nn.init.constant_(self.clf[1].weight, 0)
    for level_block in (self.down_blocks + self.up_blocks):
        for block in level_block:
            if isinstance(block, SDCascadeResBlock):
                block.channelwise[-1].weight.data *= np.sqrt(1 / sum(self.
                    config.blocks[0]))
            elif isinstance(block, SDCascadeTimestepBlock):
                nn.init.constant_(block.mapper.weight, 0)
