def init_weight(self):
    init_bn(self.bn0)
    init_layer(self.fc1)
    init_layer(self.fc_audioset)
