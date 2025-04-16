def forward(self, x):
    z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1
        )
    return self.flat(self.conv(z))
