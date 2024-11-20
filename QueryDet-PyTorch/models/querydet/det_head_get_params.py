def get_params(self):
    if not self.bn_converted:
        self._bn_convert()
    weights = [x.weight.data for x in self.subnet] + [self.pred_net.weight.data
        ]
    biases = [x.bias.data for x in self.subnet] + [self.pred_net.bias.data]
    return weights, biases
