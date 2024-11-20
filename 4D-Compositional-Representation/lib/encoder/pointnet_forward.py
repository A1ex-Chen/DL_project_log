def forward(self, x):
    batch_size, n_steps, n_pts, _ = x.shape
    if len(x.shape) == 4 and self.use_only_first_pcl:
        x = x[:, 0]
    elif len(x.shape) == 4:
        x = x.transpose(1, 2).contiguous().view(batch_size, n_pts, -1)
    net = self.fc_pos(x)
    net = self.block_0(net)
    pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
    net = torch.cat([net, pooled], dim=2)
    net = self.block_1(net)
    pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
    net = torch.cat([net, pooled], dim=2)
    net = self.block_2(net)
    pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
    net = torch.cat([net, pooled], dim=2)
    net = self.block_3(net)
    pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
    net = torch.cat([net, pooled], dim=2)
    net = self.block_4(net)
    net = self.pool(net, dim=1)
    c = self.fc_c(self.actvn(net))
    return c
