def forward(self, x):
    _, _, H_, W_ = x.shape
    Padding = False
    if (min(H_, W_) < self.window_size or H_ % self.window_size != 0 or W_ %
        self.window_size != 0):
        Padding = True
        pad_r = (self.window_size - W_ % self.window_size) % self.window_size
        pad_b = (self.window_size - H_ % self.window_size) % self.window_size
        x = F.pad(x, (0, pad_r, 0, pad_b))
    B, C, H, W = x.shape
    L = H * W
    x = x.permute(0, 2, 3, 1).contiguous().view(B, L, C)
    if self.shift_size > 0:
        attn_mask = self.create_mask(H, W).to(x.device)
    else:
        attn_mask = None
    shortcut = x
    x = x.view(B, H, W, C)
    if self.shift_size > 0:
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.
            shift_size), dims=(1, 2))
    else:
        shifted_x = x
    x_windows = window_partition_v2(shifted_x, self.window_size)
    x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
    attn_windows = self.attn(x_windows, mask=attn_mask)
    attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
    shifted_x = window_reverse_v2(attn_windows, self.window_size, H, W)
    if self.shift_size > 0:
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size),
            dims=(1, 2))
    else:
        x = shifted_x
    x = x.view(B, H * W, C)
    x = shortcut + self.drop_path(self.norm1(x))
    x = x + self.drop_path(self.norm2(self.mlp(x)))
    x = x.permute(0, 2, 1).contiguous().view(-1, C, H, W)
    if Padding:
        x = x[:, :, :H_, :W_]
    return x
