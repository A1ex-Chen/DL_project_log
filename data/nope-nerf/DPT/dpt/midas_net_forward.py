def forward(self, x):
    """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """
    layer_1 = self.pretrained.layer1(x)
    layer_2 = self.pretrained.layer2(layer_1)
    layer_3 = self.pretrained.layer3(layer_2)
    layer_4 = self.pretrained.layer4(layer_3)
    layer_1_rn = self.scratch.layer1_rn(layer_1)
    layer_2_rn = self.scratch.layer2_rn(layer_2)
    layer_3_rn = self.scratch.layer3_rn(layer_3)
    layer_4_rn = self.scratch.layer4_rn(layer_4)
    path_4 = self.scratch.refinenet4(layer_4_rn)
    path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
    path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
    path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
    out = self.scratch.output_conv(path_1)
    return torch.squeeze(out, dim=1)
