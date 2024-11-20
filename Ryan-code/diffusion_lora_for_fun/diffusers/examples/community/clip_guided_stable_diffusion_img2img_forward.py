def forward(self, pixel_values, num_cutouts):
    sideY, sideX = pixel_values.shape[2:4]
    max_size = min(sideX, sideY)
    min_size = min(sideX, sideY, self.cut_size)
    cutouts = []
    for _ in range(num_cutouts):
        size = int(torch.rand([]) ** self.cut_power * (max_size - min_size) +
            min_size)
        offsetx = torch.randint(0, sideX - size + 1, ())
        offsety = torch.randint(0, sideY - size + 1, ())
        cutout = pixel_values[:, :, offsety:offsety + size, offsetx:offsetx +
            size]
        cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
    return torch.cat(cutouts)
