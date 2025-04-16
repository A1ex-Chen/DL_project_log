def encode(self, x):
    if len(x.shape) == 4:
        batch_size, n_steps, n_pts, _ = x.shape
        x = x.transpose(1, 2).contiguous().view(batch_size, n_pts, -1)
    bsize = x.size(0)
    S = [torch.from_numpy(s).long().to(self.device) for s in self.spirals]
    D = [torch.from_numpy(s).float().to(self.device) for s in self.D]
    j = 0
    for i in range(len(self.spiral_sizes) - 1):
        x = self.conv[j](x, S[i].repeat(bsize, 1, 1))
        j += 1
        if self.filters_enc[1][i]:
            x = self.conv[j](x, S[i].repeat(bsize, 1, 1))
            j += 1
        x = torch.matmul(D[i], x)
    x = x.view(bsize, -1)
    return self.fc_latent_enc(x)
