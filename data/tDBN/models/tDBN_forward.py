def forward(self, voxel_features, coors, batch_size):
    coors = coors.int()[:, [1, 2, 3, 0]]
    ret = self.scn_input((coors.cpu(), voxel_features, batch_size))
    output = {}
    ret = self.block_input(ret)
    x0 = self.x0_in(ret)
    x1 = self.x1_in(x0)
    x2 = self.x2_in(x1)
    x3 = self.x3_in(x2)
    x2_f = self.concate2([x2, self.upsample32(x3)])
    x1_f = self.concate1([x1, self.upsample21(x2_f)])
    x0_f = self.concate0([x0, self.upsample10(x1_f)])
    x0_out = self.feature_map0(x0_f)
    x1_out = self.feature_map1(x1_f)
    x2_out = self.feature_map2(x2_f)
    x3_out = self.feature_map3(x3)
    N, C, D, H, W = x0_out.shape
    output[0] = x0_out.view(N, C * D, H, W)
    N, C, D, H, W = x1_out.shape
    output[1] = x1_out.view(N, C * D, H, W)
    N, C, D, H, W = x2_out.shape
    output[2] = x2_out.view(N, C * D, H, W)
    N, C, D, H, W = x3_out.shape
    output[3] = x3_out.view(N, C * D, H, W)
    return output
