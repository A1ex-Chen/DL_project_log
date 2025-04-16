def pad(self, *inputs):
    return [F.pad(x, self._pad, mode='replicate') for x in inputs]
