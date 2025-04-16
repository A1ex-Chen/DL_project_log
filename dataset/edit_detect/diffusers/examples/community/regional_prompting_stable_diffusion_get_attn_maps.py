def get_attn_maps(self, attn):
    height, width = self.hw
    target_tokens = self.target_tokens
    if (height, width) not in self.attnmaps_sizes:
        self.attnmaps_sizes.append((height, width))
    for b in range(self.batch):
        for t in target_tokens:
            power = self.power
            add = attn[b, :, :, t[0]:t[0] + len(t)] ** power * (self.
                attnmaps_sizes.index((height, width)) + 1)
            add = torch.sum(add, dim=2)
            key = f'{t}-{b}'
            if key not in self.attnmaps:
                self.attnmaps[key] = add
            else:
                if self.attnmaps[key].shape[1] != add.shape[1]:
                    add = add.view(8, height, width)
                    add = FF.resize(add, self.attnmaps_sizes[0], antialias=None
                        )
                    add = add.reshape_as(self.attnmaps[key])
                self.attnmaps[key] = self.attnmaps[key] + add
