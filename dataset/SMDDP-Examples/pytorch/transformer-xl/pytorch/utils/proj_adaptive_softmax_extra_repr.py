def extra_repr(self):
    child_lines = []
    for k, p in self._parameters.items():
        if p is not None:
            size_str = 'x'.join(str(size) for size in p.size())
            device_str = '' if not p.is_cuda else ' (GPU {})'.format(p.
                get_device())
            parastr = 'Parameter containing: [{} of size {}{}]'.format(torch
                .typename(p), size_str, device_str)
            child_lines.append('  (' + str(k) + '): ' + parastr)
    tmpstr = '\n'.join(child_lines)
    return tmpstr
