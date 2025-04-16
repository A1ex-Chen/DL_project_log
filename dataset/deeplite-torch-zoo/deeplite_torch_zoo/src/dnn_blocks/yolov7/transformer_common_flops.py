def flops(self):
    flops = 0
    H, W = self.input_resolution
    flops += self.dim * H * W
    nW = H * W / self.window_size / self.window_size
    flops += nW * self.attn.flops(self.window_size * self.window_size)
    flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
    flops += self.dim * H * W
    return flops
