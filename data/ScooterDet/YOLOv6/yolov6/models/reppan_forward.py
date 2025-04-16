def forward(self, input):
    x2, x1, x0 = input
    fpn_out0 = self.reduce_layer0(x0)
    x1 = self.reduce_layer1(x1)
    x2 = self.reduce_layer2(x2)
    upsample_feat0 = self.upsample0(fpn_out0)
    f_concat_layer0 = torch.cat([upsample_feat0, x1], 1)
    f_out1 = self.Csp_p4(f_concat_layer0)
    upsample_feat1 = self.upsample1(f_out1)
    f_concat_layer1 = torch.cat([upsample_feat1, x2], 1)
    pan_out3 = self.Csp_p3(f_concat_layer1)
    down_feat1 = self.downsample2(pan_out3)
    p_concat_layer1 = torch.cat([down_feat1, f_out1], 1)
    pan_out2 = self.Csp_n3(p_concat_layer1)
    down_feat0 = self.downsample1(pan_out2)
    p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
    pan_out1 = self.Csp_n4(p_concat_layer2)
    top_features = self.p6_conv_1(fpn_out0)
    pan_out0 = top_features + self.p6_conv_2(pan_out1)
    outputs = [pan_out3, pan_out2, pan_out1, pan_out0]
    return outputs
