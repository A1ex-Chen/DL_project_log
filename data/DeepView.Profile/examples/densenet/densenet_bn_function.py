def bn_function(self, inputs):
    concated_features = torch.cat(inputs, 1)
    bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
    return bottleneck_output
