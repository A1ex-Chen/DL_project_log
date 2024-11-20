def forward(self, x):
    identity = x
    x = self.compress_conv_bn_act1(x)
    x = self.c_shuffle(x)
    x = self.dw_conv_bn2(x)
    x = self.expand_conv_bn3(x)
    if self.downsample:
        identity = self.avgpool(identity)
        x = torch.cat((x, identity), dim=1)
    else:
        x = x + identity
    x = self.activ(x)
    return x
