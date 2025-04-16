def _convert_to_dot_list(self, opts):
    if opts is None:
        opts = []
    if len(opts) == 0:
        return opts
    has_equal = opts[0].find('=') != -1
    if has_equal:
        return opts
    return [(opt + '=' + value) for opt, value in zip(opts[0::2], opts[1::2])]
