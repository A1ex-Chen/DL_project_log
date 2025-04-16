@torch.no_grad()
def fuse(self):
    """
        Fuses the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
    conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
    conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)
    conv_w = conv.weight
    conv_b = conv.bias
    conv1_w = conv1.weight
    conv1_b = conv1.bias
    conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])
    final_conv_w = conv_w + conv1_w
    final_conv_b = conv_b + conv1_b
    conv.weight.data.copy_(final_conv_w)
    conv.bias.data.copy_(final_conv_b)
    self.conv = conv
    del self.conv1
