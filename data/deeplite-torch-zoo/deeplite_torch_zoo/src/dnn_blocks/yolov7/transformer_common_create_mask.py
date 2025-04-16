def create_mask(self, H, W):
    img_mask = torch.zeros((1, H, W, 1))
    h_slices = slice(0, -self.window_size), slice(-self.window_size, -self.
        shift_size), slice(-self.shift_size, None)
    w_slices = slice(0, -self.window_size), slice(-self.window_size, -self.
        shift_size), slice(-self.shift_size, None)
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = window_partition(img_mask, self.window_size)
    mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)
        ).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask
