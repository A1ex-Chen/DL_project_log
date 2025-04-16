@property
def last_features_module(self) ->nn.Module:
    return self.net._conv_head
