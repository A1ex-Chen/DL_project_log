def forward(self, x):
    branch1x1 = self.branch1x1(x)
    branch3x3 = self.branch3x3_1(x)
    branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
    branch3x3 = torch.cat(branch3x3, 1)
    branch3x3dbl = self.branch3x3dbl_1(x)
    branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
    branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.
        branch3x3dbl_3b(branch3x3dbl)]
    branch3x3dbl = torch.cat(branch3x3dbl, 1)
    branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch_pool = self.branch_pool(branch_pool)
    outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
    return torch.cat(outputs, 1)
