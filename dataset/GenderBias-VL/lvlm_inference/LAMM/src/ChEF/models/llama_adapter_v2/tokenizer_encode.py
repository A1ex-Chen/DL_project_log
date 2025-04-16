def encode(self, s: str, bos: bool, eos: bool) ->List[int]:
    assert type(s) is str
    t = self.sp_model.encode(s)
    if bos:
        t = [self.bos_id] + t
    if eos:
        t = t + [self.eos_id]
    return t
