def _forward(self, x):
    b, c, *spatial = x.shape
    x = x.reshape(b, c, -1).contiguous()
    qkv = self.qkv(self.norm(x)).contiguous()
    h = self.attention(qkv).contiguous()
    h = self.proj_out(h).contiguous()
    return (x + h).reshape(b, c, *spatial).contiguous()
