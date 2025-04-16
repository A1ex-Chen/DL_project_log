def forward(self, image_batch: Tensor) ->Tuple[Tensor, Tensor, Tensor,
    Tensor, Tensor]:
    c1_batch = self.conv1(image_batch)
    c2_batch = self.conv2(c1_batch)
    c3_batch = self.conv3(c2_batch)
    c4_batch = self.conv4(c3_batch)
    c5_batch = self.conv5(c4_batch)
    x_batch = {'c2': c2_batch, 'c3': c3_batch, 'c4': c4_batch, 'c5': c5_batch}
    x_out_batch = self.fpn(x_batch)
    p2_batch = x_out_batch['c2']
    p3_batch = x_out_batch['c3']
    p4_batch = x_out_batch['c4']
    p5_batch = x_out_batch['c5']
    p6_batch = x_out_batch['pool']
    return p2_batch, p3_batch, p4_batch, p5_batch, p6_batch
