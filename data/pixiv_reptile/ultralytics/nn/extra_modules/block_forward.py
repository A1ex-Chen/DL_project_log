def forward(self, x):
    if self.fusion in ['weight', 'adaptive']:
        for i in range(len(x)):
            x[i] = self.fusion_conv[i](x[i])
    if self.fusion == 'weight':
        return torch.sum(torch.stack(x, dim=0), dim=0)
    elif self.fusion == 'adaptive':
        fusion = torch.softmax(self.fusion_adaptive(torch.cat(x, dim=1)), dim=1
            )
        x_weight = torch.split(fusion, [1] * len(x), dim=1)
        return torch.sum(torch.stack([(x_weight[i] * x[i]) for i in range(
            len(x))], dim=0), dim=0)
    elif self.fusion == 'concat':
        return torch.cat(x, dim=1)
    elif self.fusion == 'bifpn':
        fusion_weight = self.relu(self.fusion_weight.clone())
        fusion_weight = fusion_weight / torch.sum(fusion_weight, dim=0)
        return torch.sum(torch.stack([(fusion_weight[i] * x[i]) for i in
            range(len(x))], dim=0), dim=0)
    elif self.fusion == 'SDI':
        return self.SDI(x)
