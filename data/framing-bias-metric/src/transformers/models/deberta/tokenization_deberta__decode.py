def _decode(self, x: str) ->str:
    return self.bpe.decode(map(int, x.split()))
