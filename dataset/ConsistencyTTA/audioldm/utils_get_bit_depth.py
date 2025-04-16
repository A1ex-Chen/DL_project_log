def get_bit_depth(fname):
    with contextlib.closing(wave.open(fname, 'r')) as f:
        bit_depth = f.getsampwidth() * 8
        return bit_depth
