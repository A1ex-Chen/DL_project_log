def _forward_eval(self, features):
    if not self.bn_converted:
        self._bn_convert()
    for i in range(self.num_convs):
        features = [F.relu(self.subnet[i](x)) for x in features]
    preds = [self.pred_net(x) for x in features]
    return preds
