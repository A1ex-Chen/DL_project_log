def _init_weights(self, m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    for mapper in self.effnet_mappers:
        if mapper is not None:
            nn.init.normal_(mapper.weight, std=0.02)
    nn.init.normal_(self.clip_mapper.weight, std=0.02)
    nn.init.xavier_uniform_(self.embedding[1].weight, 0.02)
    nn.init.constant_(self.clf[1].weight, 0)
    for level_block in (self.down_blocks + self.up_blocks):
        for block in level_block:
            if isinstance(block, ResBlockStageB):
                block.channelwise[-1].weight.data *= np.sqrt(1 / sum(self.
                    config.blocks))
            elif isinstance(block, TimestepBlock):
                nn.init.constant_(block.mapper.weight, 0)
