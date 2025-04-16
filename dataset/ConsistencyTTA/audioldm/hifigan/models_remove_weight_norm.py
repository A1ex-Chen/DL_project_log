def remove_weight_norm(self):
    for l in self.ups:
        remove_weight_norm(l)
    for l in self.resblocks:
        l.remove_weight_norm()
    remove_weight_norm(self.conv_pre)
    remove_weight_norm(self.conv_post)
