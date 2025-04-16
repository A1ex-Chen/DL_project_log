def forward(self, x):
    kernel = x.size()[2], x.size()[3]
    avg_pool = F.avg_pool2d(x, kernel)
    max_pool = F.max_pool2d(x, kernel)
    avg_pool = avg_pool.view(avg_pool.size()[0], -1)
    max_pool = max_pool.view(max_pool.size()[0], -1)
    avg_pool_bck = self.bottleneck(avg_pool)
    max_pool_bck = self.bottleneck(max_pool)
    pool_sum = avg_pool_bck + max_pool_bck
    sig_pool = torch.sigmoid(pool_sum)
    sig_pool = sig_pool.unsqueeze(2).unsqueeze(3)
    out = sig_pool.repeat(1, 1, kernel[0], kernel[1])
    return out
