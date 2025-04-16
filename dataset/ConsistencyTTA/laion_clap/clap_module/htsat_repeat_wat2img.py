def repeat_wat2img(self, x, cur_pos):
    B, C, T, F = x.shape
    target_T = int(self.spec_size * self.freq_ratio)
    target_F = self.spec_size // self.freq_ratio
    assert T <= target_T and F <= target_F, 'the wav size should less than or equal to the swin input size'
    if T < target_T:
        x = nn.functional.interpolate(x, (target_T, x.shape[3]), mode=
            'bicubic', align_corners=True)
    if F < target_F:
        x = nn.functional.interpolate(x, (x.shape[2], target_F), mode=
            'bicubic', align_corners=True)
    x = x.permute(0, 1, 3, 2).contiguous()
    x = x[:, :, :, cur_pos:cur_pos + self.spec_size]
    x = x.repeat(repeats=(1, 1, 4, 1))
    return x
