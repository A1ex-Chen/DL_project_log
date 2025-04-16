def _chunk_tensor(self, t: Iterable, chunk_size: int) ->List[Iterable]:
    return [t[i:i + chunk_size] for i in range(0, len(t), chunk_size)]
