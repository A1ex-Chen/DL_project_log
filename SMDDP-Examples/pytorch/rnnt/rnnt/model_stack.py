def stack(self, x):
    x = x.transpose(0, 1)
    T = x.size(1)
    padded = torch.nn.functional.pad(x, (0, 0, 0, (self.factor - T % self.
        factor) % self.factor))
    B, T, H = padded.size()
    x = padded.reshape(B, T // self.factor, -1)
    x = x.transpose(0, 1)
    return x
