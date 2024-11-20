def _forward_train(self, features):
    for i in range(self.num_convs):
        features = [self.subnet[i](x) for x in features]
        features = self.bns[i](features)
        features = [F.relu(x) for x in features]
    preds = [self.pred_net(x) for x in features]
    return preds
