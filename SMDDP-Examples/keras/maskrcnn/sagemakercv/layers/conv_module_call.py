def call(self, x, training=None):
    x = self.conv(x)
    if self.norm:
        x = self.norm(x, training=training)
    if self.act:
        x = self.act(x)
    return x
