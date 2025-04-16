def forward_resblock(self, x, layers):
    for l in layers:
        x = l(x)
    return x
