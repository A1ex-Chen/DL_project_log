def init_weights(self):
    init_layer(self.att)
    init_layer(self.cla)
    init_bn(self.bn_att)
