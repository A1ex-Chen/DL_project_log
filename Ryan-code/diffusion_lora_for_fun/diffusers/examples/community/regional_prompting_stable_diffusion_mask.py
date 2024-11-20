def mask(input):
    out = torch.multiply(input, self.attnmasks[h, w])
    for b in range(batch):
        for r in range(1, regions):
            out[b] = out[b] + out[r * batch + b]
    return out
