def forward_features(self, x):
    x = self.model.forward_features(x)
    W = H = self.img_size // self.patch_size
    T = self.num_frames
    cls_tokens = x[:, 0, :].unsqueeze(1)
    other_tokens = x[:, 1:, :]
    x = rearrange(other_tokens, 'b (h w t) m -> b t (h w) m', h=H, w=W, t=T)
    x = torch.mean(x, dim=1)
    x = torch.cat((cls_tokens, x), dim=1)
    return x
