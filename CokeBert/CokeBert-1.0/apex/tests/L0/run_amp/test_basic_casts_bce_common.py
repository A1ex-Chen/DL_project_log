def bce_common(self, assertion):
    shape = self.b, self.h
    target = torch.rand(shape)
    mod = nn.BCELoss()
    m = lambda x: mod(x, target)
    f = ft.partial(F.binary_cross_entropy, target=target)
    for fn in [m, f]:
        x = torch.rand(shape, dtype=torch.half)
        assertion(fn, x)
