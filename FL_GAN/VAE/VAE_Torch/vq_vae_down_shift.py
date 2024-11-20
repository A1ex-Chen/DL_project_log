def down_shift(self, x):
    x = x[:, :, :-1, :]
    pad = nn.ZeroPad2d((0, 0, 1, 0))
    return pad(x)
