def _encode(self, x: str) ->str:
    return ' '.join(map(str, self.bpe.encode(x)))
