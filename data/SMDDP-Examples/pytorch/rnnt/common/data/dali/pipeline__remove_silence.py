def _remove_silence(self, inp):
    begin, length = self.get_nonsilent_region(inp)
    out = self.trim_silence(inp, self.to_float(begin), self.to_float(length))
    return out
