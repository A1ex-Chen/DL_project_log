def forward(self, features):
    if self.training:
        return self._forward_train(features)
    else:
        return self._forward_eval(features)
