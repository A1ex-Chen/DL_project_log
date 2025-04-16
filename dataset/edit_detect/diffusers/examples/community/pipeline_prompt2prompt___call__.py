def __call__(self, x_t, attention_store):
    k = 1
    maps = attention_store['down_cross'][2:4] + attention_store['up_cross'][:3]
    maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, self.
        max_num_words) for item in maps]
    maps = torch.cat(maps, dim=1)
    maps = (maps * self.alpha_layers).sum(-1).mean(1)
    mask = F.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
    mask = F.interpolate(mask, size=x_t.shape[2:])
    mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
    mask = mask.gt(self.threshold)
    mask = (mask[:1] + mask[1:]).float()
    x_t = x_t[:1] + mask * (x_t - x_t[:1])
    return x_t
