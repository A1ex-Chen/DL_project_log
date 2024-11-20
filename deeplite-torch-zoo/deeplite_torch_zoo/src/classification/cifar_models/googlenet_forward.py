def forward(self, x):
    out = self.pre_layers(x)
    out = self.a3(out)
    out = self.b3(out)
    out = self.maxpool(out)
    out = self.a4(out)
    out = self.b4(out)
    out = self.c4(out)
    out = self.d4(out)
    out = self.e4(out)
    out = self.maxpool(out)
    out = self.a5(out)
    out = self.b5(out)
    out = self.avgpool(out)
    out = out.view(out.size(0), -1)
    out = self.linear(out)
    return out