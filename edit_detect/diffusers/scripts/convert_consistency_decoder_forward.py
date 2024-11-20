def forward(self, x, t, features) ->torch.Tensor:
    converted = hasattr(self, 'converted') and self.converted
    x = torch.cat([x, F.upsample_nearest(features, scale_factor=8)], dim=1)
    if converted:
        t = self.time_embedding(self.time_proj(t))
    else:
        t = self.embed_time(t)
    x = self.embed_image(x)
    skips = [x]
    for i, down in enumerate(self.down):
        if converted and i in [0, 1, 2, 3]:
            x, skips_ = down(x, t)
            for skip in skips_:
                skips.append(skip)
        else:
            for block in down:
                x = block(x, t)
                skips.append(x)
        print(x.float().abs().sum())
    if converted:
        x = self.mid(x, t)
    else:
        for i in range(2):
            x = self.mid[i](x, t)
    print(x.float().abs().sum())
    for i, up in enumerate(self.up[::-1]):
        if converted and i in [0, 1, 2, 3]:
            skip_4 = skips.pop()
            skip_3 = skips.pop()
            skip_2 = skips.pop()
            skip_1 = skips.pop()
            skips_ = skip_1, skip_2, skip_3, skip_4
            x = up(x, skips_, t)
        else:
            for block in up:
                if isinstance(block, ConvResblock):
                    x = torch.concat([x, skips.pop()], dim=1)
                x = block(x, t)
    return self.output(x)
