@staticmethod
def _split(values, sizes):
    if len(sizes):
        expected_len = sum(sizes)
        assert len(values
            ) == expected_len, f'Values has length {len(values)} but expect length {expected_len}.'
    ret = []
    for k in range(len(sizes)):
        begin, end = sum(sizes[:k]), sum(sizes[:k + 1])
        ret.append(values[begin:end])
    return ret
