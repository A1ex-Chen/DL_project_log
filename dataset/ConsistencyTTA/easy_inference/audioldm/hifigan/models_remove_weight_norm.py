def remove_weight_norm(self):
    for l in self.ups:
        remove_parametrizations(l, 'weight')
    for l in self.resblocks:
        l.remove_weight_norm()
    remove_parametrizations(self.conv_pre, 'weight')
    remove_parametrizations(self.conv_post, 'weight')
