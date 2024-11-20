def forward(self, x, im_mask=None):
    res = super().forward(x)
    if im_mask is not None:
        if torch.sum(im_mask) > 0:
            part_x = x[im_mask]
            res[im_mask] += self.Plora_B(self.Plora_A(self.lora_dropout(
                part_x))) * self.lora_scaling
        else:
            part_x = x[:, :1]
            res[:, :1] += self.Plora_B(self.Plora_A(self.lora_dropout(part_x))
                ) * 0
    return res
