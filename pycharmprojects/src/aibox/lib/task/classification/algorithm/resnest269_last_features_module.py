@property
def last_features_module(self) ->nn.Module:
    return self.net.layer4