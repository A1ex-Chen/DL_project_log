def as_sequential(self):
    layers = [self.conv_stem, self.bn1, self.act1]
    layers.extend(self.blocks)
    layers.extend([self.global_pool, self.conv_head, self.act2, nn.Flatten(
        ), nn.Dropout(self.drop_rate), self.classifier])
    return nn.Sequential(*layers)
