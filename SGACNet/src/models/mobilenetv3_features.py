def features(self, x):
    x = self.conv_stem(x)
    x = self.bn1(x)
    x = self.act1(x)
    x = self.blocks(x)
    x = self.global_pool(x)
    x = self.conv_head(x)
    x = self.act2(x)
    return x
