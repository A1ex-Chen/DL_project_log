def forward(self, x):
    x = self.conv_stem(x)
    x = self.bn1(x)
    x = self.act1(x)
    x = self.blocks(x)
    x = self.global_pool(x)
    x = self.conv_head(x)
    x = self.act2(x)
    x = x.view(x.size(0), -1)
    if self.dropout > 0.0:
        x = F.dropout(x, p=self.dropout, training=self.training)
    x = self.classifier(x)
    return x
