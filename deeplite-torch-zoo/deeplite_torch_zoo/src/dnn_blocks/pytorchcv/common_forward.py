def forward(self, x):
    return channel_shuffle(x, self.groups)
