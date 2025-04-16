@functools.lru_cache(maxsize=None)
def _can_match(self, f, b):
    if 'aten' in f and 'Backward0' in b:
        raw_f = f[len('aten::'):].lower().replace('_', '')
        raw_b = b[:-len('Backward0')].lower()
        return (raw_f == raw_b or raw_f == 'transpose' and raw_b == 't' or 
            raw_f == 't' and raw_b == 'transpose')
    return False
