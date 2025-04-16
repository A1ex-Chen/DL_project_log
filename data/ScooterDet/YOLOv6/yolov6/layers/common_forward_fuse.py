def forward_fuse(self, x):
    x = self.act_1(self.conv_dw_1(x))
    x = self.act_2(self.conv_pw_1(x))
    return x
