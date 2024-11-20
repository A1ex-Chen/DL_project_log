def forward(self, t, y):
    net = self.a(y ** 3)
    net = self.b(net)
    net = self.c(net)
    return net
